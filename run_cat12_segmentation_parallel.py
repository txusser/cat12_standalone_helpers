#!/usr/bin/env python3
"""
Dynamic parallel CAT12 processing script.

Runs CAT12 Docker container on prepared NIfTI images with dynamic scaling:
- Monitors running CAT12 containers and keeps below max limit
- Checks available memory before spawning new jobs
- Skips already processed images (having mwp1*.nii and cat_*.xml files)
"""

import os
import subprocess
import argparse
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ProcessingResult:
    """Result of a CAT12 processing job."""
    subject: str
    nii_file: str
    success: bool = False
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class DynamicConfig:
    """Configuration for dynamic scaling."""
    max_concurrent: int = 16
    min_free_memory_gb: float = 8.0
    poll_interval_seconds: float = 60.0
    docker_image: str = "jhuguetn/cat12"


def get_free_memory_gb() -> float:
    """
    Get available free memory in GB.
    
    Works on Linux by reading /proc/meminfo.
    Falls back to psutil if available, otherwise returns a large value.
    """
    # Try reading from /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Value is in kB
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)  # Convert to GB
    except FileNotFoundError:
        pass
    
    # Try using psutil if available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    
    # Fallback: assume enough memory (for macOS without psutil)
    print("Warning: Cannot determine free memory, assuming sufficient")
    return 999.0


def count_running_cat12_containers(docker_image: str = "jhuguetn/cat12") -> int:
    """
    Count the number of currently running CAT12 Docker containers.
    
    Args:
        docker_image: The Docker image name to filter by
    
    Returns:
        Number of running containers using the specified image
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"ancestor={docker_image}", "-q"],
            capture_output=True,
            text=True,
            check=True
        )
        # Count non-empty lines
        containers = [c for c in result.stdout.strip().split("\n") if c]
        return len(containers)
    except subprocess.CalledProcessError:
        return 0
    except Exception as e:
        print(f"Warning: Error counting containers: {e}")
        return 0


def is_already_processed(cat12_path: Path, t1_name: str) -> bool:
    """
    Check if an image has already been processed by CAT12.
    
    Args:
        cat12_path: Path to the cat12 folder
        t1_name: Base name of the T1 image (without .nii extension)
    
    Returns:
        True if both mwp1*.nii and cat_*.xml files exist
    """
    mwp1_file = cat12_path / f"mwp1{t1_name}.nii"
    cat_xml_file = cat12_path / f"cat_{t1_name}.xml"
    
    return mwp1_file.exists() and cat_xml_file.exists()


def find_images_to_process(masterdata_dir: str, verbose: bool = True) -> list[tuple[Path, str]]:
    """
    Find all prepared images that need CAT12 processing.
    
    Args:
        masterdata_dir: Root directory containing subject folders
        verbose: Print status messages for each folder
    
    Returns:
        List of tuples (subject_folder_path, nii_filename) for images to process
    """
    to_process = []
    masterdata_path = Path(masterdata_dir)
    skipped_processed = 0
    skipped_no_cat12 = 0
    skipped_no_nii = 0
    
    for img_folder in sorted(masterdata_path.iterdir()):
        if not img_folder.is_dir():
            continue
            
        cat12_path = img_folder / "cat12"
        
        if not cat12_path.exists():
            if verbose:
                print(f"Skipping {img_folder.name}: No cat12 folder")
            skipped_no_cat12 += 1
            continue
        
        # Find t1_*.nii file in cat12 folder (unzipped, ready for processing)
        nii_files = list(cat12_path.glob("t1_*.nii"))
        
        if not nii_files:
            if verbose:
                print(f"Skipping {img_folder.name}: No t1_*.nii file")
            skipped_no_nii += 1
            continue
        
        nii_file = nii_files[0]
        t1_name = nii_file.stem  # filename without .nii extension
        
        if is_already_processed(cat12_path, t1_name):
            skipped_processed += 1
            continue
        
        to_process.append((img_folder, nii_file.name))
    
    print(f"\nScan summary:")
    print(f"  - To process: {len(to_process)}")
    print(f"  - Already processed: {skipped_processed}")
    print(f"  - No cat12 folder: {skipped_no_cat12}")
    print(f"  - No .nii file: {skipped_no_nii}")
    
    return to_process


async def run_cat12_async(
    subject_folder: Path, 
    nii_filename: str, 
    config: DynamicConfig,
    dry_run: bool = False
) -> ProcessingResult:
    """
    Run CAT12 Docker container asynchronously.
    
    Args:
        subject_folder: Path to the subject folder
        nii_filename: Name of the NIfTI file to process
        config: Dynamic configuration
        dry_run: If True, simulate without executing
    
    Returns:
        ProcessingResult with job outcome
    """
    result = ProcessingResult(
        subject=subject_folder.name,
        nii_file=nii_filename,
        start_time=datetime.now()
    )
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{subject_folder}:/data",
        config.docker_image,
        "-b", "/opt/spm/standalone/cat_standalone_segment.m",
        f"/data/cat12/{nii_filename}"
    ]
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        await asyncio.sleep(0.5)  # Simulate some work
        result.success = True
        result.end_time = datetime.now()
        return result
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            result.success = True
        else:
            result.error = f"Return code {process.returncode}: {stderr.decode()[:200]}"
            
    except Exception as e:
        result.error = str(e)
    
    result.end_time = datetime.now()
    return result


def can_start_new_job(config: DynamicConfig) -> tuple[bool, str]:
    """
    Check if conditions allow starting a new job.
    
    Args:
        config: Dynamic configuration with thresholds
    
    Returns:
        Tuple of (can_start, reason_if_not)
    """
    # Check running containers
    running = count_running_cat12_containers(config.docker_image)
    if running >= config.max_concurrent:
        return False, f"Max containers reached ({running}/{config.max_concurrent})"
    
    # Check free memory
    free_mem = get_free_memory_gb()
    if free_mem < config.min_free_memory_gb:
        return False, f"Low memory ({free_mem:.1f}GB < {config.min_free_memory_gb}GB)"
    
    return True, ""


async def dynamic_scheduler(
    images_to_process: list[tuple[Path, str]],
    config: DynamicConfig,
    dry_run: bool = False
) -> list[ProcessingResult]:
    """
    Dynamically schedule CAT12 jobs based on system resources.
    
    Args:
        images_to_process: List of (subject_folder, nii_filename) tuples
        config: Dynamic configuration
        dry_run: If True, simulate without executing
    
    Returns:
        List of ProcessingResult objects
    """
    pending = list(images_to_process)
    running_tasks: dict[asyncio.Task, str] = {}
    completed_results: list[ProcessingResult] = []
    
    print(f"\nDynamic scheduler started:")
    print(f"  - Max concurrent jobs: {config.max_concurrent}")
    print(f"  - Min free memory: {config.min_free_memory_gb} GB")
    print(f"  - Poll interval: {config.poll_interval_seconds}s")
    print(f"  - Total jobs: {len(pending)}\n")
    
    while pending or running_tasks:
        # Check for completed tasks
        if running_tasks:
            done_tasks = [t for t in running_tasks if t.done()]
            for task in done_tasks:
                subject_name = running_tasks.pop(task)
                try:
                    result = task.result()
                    completed_results.append(result)
                    status = "✓" if result.success else "✗"
                    duration = ""
                    if result.start_time and result.end_time:
                        dur = (result.end_time - result.start_time).total_seconds()
                        duration = f" ({dur:.0f}s)"
                    print(f"{status} Completed: {subject_name}{duration}")
                    if not result.success and result.error:
                        print(f"   Error: {result.error[:100]}")
                except Exception as e:
                    print(f"✗ Exception for {subject_name}: {e}")
                    completed_results.append(ProcessingResult(
                        subject=subject_name,
                        nii_file="",
                        success=False,
                        error=str(e)
                    ))
        
        # Try to start new jobs
        while pending:
            can_start, reason = can_start_new_job(config)
            if not can_start:
                break
            
            subject_folder, nii_file = pending.pop(0)
            task = asyncio.create_task(
                run_cat12_async(subject_folder, nii_file, config, dry_run)
            )
            running_tasks[task] = subject_folder.name
            
            running = count_running_cat12_containers(config.docker_image)
            free_mem = get_free_memory_gb()
            print(f"▶ Started: {subject_folder.name} "
                  f"[running: {running + 1}, pending: {len(pending)}, "
                  f"mem: {free_mem:.1f}GB]")
        
        # Status update
        if running_tasks:
            running = count_running_cat12_containers(config.docker_image)
            free_mem = get_free_memory_gb()
            can_start, reason = can_start_new_job(config)
            if not can_start and pending:
                print(f"⏸ Waiting: {reason} "
                      f"[running: {len(running_tasks)}, pending: {len(pending)}]")
        
        # Wait before next poll
        if running_tasks or pending:
            await asyncio.sleep(config.poll_interval_seconds)
    
    return completed_results


def main():
    parser = argparse.ArgumentParser(
        description="Run CAT12 Docker processing with dynamic resource-based scaling"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/data/ADNI/T1",
        help="Master data directory containing subject folders"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=16,
        help="Maximum number of concurrent Docker containers (default: 16)"
    )
    parser.add_argument(
        "--min-memory",
        type=float,
        default=8.0,
        help="Minimum free memory in GB to start new jobs (default: 8.0)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between resource checks (default: 5.0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without running Docker commands"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list images to process, don't execute anything"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output during scan"
    )
    
    args = parser.parse_args()
    
    config = DynamicConfig(
        max_concurrent=args.max_concurrent,
        min_free_memory_gb=args.min_memory,
        poll_interval_seconds=args.poll_interval
    )
    
    print(f"Scanning {args.data_dir} for images to process...")
    images_to_process = find_images_to_process(args.data_dir, verbose=args.verbose)
    
    print(f"\n{'='*60}")
    print(f"Found {len(images_to_process)} images to process")
    print(f"{'='*60}")
    
    if not images_to_process:
        print("No images need processing. Exiting.")
        return
    
    if args.list_only:
        print("\nImages to process:")
        for subject_folder, nii_file in images_to_process:
            print(f"  - {subject_folder.name}/{nii_file}")
        return
    
    # Show current system status
    running = count_running_cat12_containers(config.docker_image)
    free_mem = get_free_memory_gb()
    print(f"\nCurrent system status:")
    print(f"  - Running CAT12 containers: {running}")
    print(f"  - Free memory: {free_mem:.1f} GB")
    
    # Run with dynamic scheduling
    results = asyncio.run(
        dynamic_scheduler(images_to_process, config, args.dry_run)
    )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed subjects:")
        for r in results:
            if not r.success:
                print(f"  - {r.subject}: {r.error or 'Unknown error'}")


if __name__ == "__main__":
    main()
