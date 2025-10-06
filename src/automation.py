"""
Automation module for continual learning and automatic retraining.

This module implements file system monitoring and automatic retraining
capabilities using the watchdog library. It provides both real-time
monitoring and scheduled retraining options.
"""

import os
import time
import logging
import threading
import argparse
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timedelta
import json
import queue
import signal
import sys

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
import numpy as np

from .train import ContinualLearningTrainer
from .data_loader import ElementalDataLoader
from .model import ModelManager, ElementDistributionModel
from .config import (
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, INCREMENTAL_MODE,
    RETRAIN_ON_NEW_DATA, WATCH_RECURSIVELY, FILE_PATTERNS,
    IGNORE_PATTERNS, LOG_FILE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetrainingEventHandler(FileSystemEventHandler):
    """
    Event handler for file system changes that trigger retraining.
    """
    
    def __init__(
        self,
        trainer: ContinualLearningTrainer,
        retraining_queue: queue.Queue,
        processed_files: Dict[str, float],
        debounce_time: float = 5.0
    ):
        """
        Initialize the event handler.
        
        Args:
            trainer: ContinualLearningTrainer instance
            retraining_queue: Queue for retraining requests
            processed_files: Dictionary to track processed files
            debounce_time: Time to wait before processing files (seconds)
        """
        super().__init__()
        self.trainer = trainer
        self.retraining_queue = retraining_queue
        self.processed_files = processed_files
        self.debounce_time = debounce_time
        self.last_event_time = {}
        
        logger.info(f"Initialized RetrainingEventHandler with debounce time: {debounce_time}s")
    
    def _should_process_file(self, file_path: str) -> bool:
        """
        Check if a file should be processed for retraining.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be processed
        """
        file_path = Path(file_path)
        
        # Check file extension
        if not any(file_path.suffix.lower() in pattern.replace('*', '') for pattern in FILE_PATTERNS):
            return False
        
        # Check ignore patterns
        for ignore_pattern in IGNORE_PATTERNS:
            if ignore_pattern.replace('*', '') in file_path.name:
                return False
        
        # Check if file was recently processed
        current_time = time.time()
        if file_path.name in self.processed_files:
            last_processed = self.processed_files[file_path.name]
            if current_time - last_processed < self.debounce_time:
                return False
        
        # Check if file is large enough (avoid processing empty or incomplete files)
        try:
            if file_path.stat().st_size < 100:  # Less than 100 bytes
                return False
        except OSError:
            return False
        
        return True
    
    def _debounce_retraining(self, file_path: str):
        """
        Debounce retraining requests to avoid multiple rapid retraining.
        
        Args:
            file_path: Path to the changed file
        """
        file_name = Path(file_path).name
        current_time = time.time()
        
        # Update last event time
        self.last_event_time[file_name] = current_time
        
        # Wait for debounce period
        def delayed_retraining():
            time.sleep(self.debounce_time)
            
            # Check if this is still the most recent event for this file
            if (file_name in self.last_event_time and 
                self.last_event_time[file_name] == current_time):
                
                if self._should_process_file(file_path):
                    self.retraining_queue.put({
                        'type': 'retrain',
                        'file_path': file_path,
                        'timestamp': current_time,
                        'reason': 'new_file'
                    })
                    
                    self.processed_files[file_name] = current_time
                    logger.info(f"Queued retraining for file: {file_path}")
        
        # Start debounce timer in separate thread
        threading.Thread(target=delayed_retraining, daemon=True).start()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            logger.info(f"File created: {event.src_path}")
            self._debounce_retraining(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            logger.info(f"File modified: {event.src_path}")
            self._debounce_retraining(event.src_path)

class AutomationManager:
    """
    Main automation manager for continual learning.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        outputs_dir: Optional[Path] = None,
        incremental_mode: bool = INCREMENTAL_MODE
    ):
        """
        Initialize the automation manager.
        
        Args:
            data_dir: Directory to monitor for new data
            models_dir: Directory for model storage
            outputs_dir: Directory for output files
            incremental_mode: Whether to use incremental learning
        """
        self.data_dir = data_dir or DATA_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.outputs_dir = outputs_dir or OUTPUTS_DIR
        self.incremental_mode = incremental_mode
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.trainer = None
        self.data_loader = ElementalDataLoader(self.data_dir)
        self.retraining_queue = queue.Queue()
        self.processed_files = {}
        self.observer = None
        self.is_running = False
        
        # Load existing model if available
        self._load_existing_model()
        
        # Statistics
        self.stats = {
            'retraining_count': 0,
            'last_retraining': None,
            'total_training_time': 0.0,
            'processed_files': 0
        }
        
        logger.info(f"Initialized AutomationManager:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Incremental mode: {incremental_mode}")
    
    def _load_existing_model(self):
        """Load existing model if available."""
        best_model_path = self.models_dir / "best_model.pt"
        
        if best_model_path.exists():
            try:
                model = ElementDistributionModel()
                model_manager = ModelManager(model, self.models_dir)
                model_manager.load_checkpoint(best_model_path)
                
                self.trainer = ContinualLearningTrainer(model)
                logger.info("Loaded existing model for continual learning")
                
            except Exception as e:
                logger.error(f"Failed to load existing model: {e}")
                self.trainer = ContinualLearningTrainer()
        else:
            logger.info("No existing model found, will train from scratch")
            self.trainer = ContinualLearningTrainer()
    
    def _process_retraining_queue(self):
        """Process retraining requests from the queue."""
        while self.is_running:
            try:
                # Get retraining request with timeout
                request = self.retraining_queue.get(timeout=1.0)
                
                if request['type'] == 'retrain':
                    self._perform_retraining(request)
                
                self.retraining_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing retraining queue: {e}")
    
    def _perform_retraining(self, request: Dict[str, Any]):
        """
        Perform retraining based on the request.
        
        Args:
            request: Retraining request dictionary
        """
        file_path = request['file_path']
        timestamp = request['timestamp']
        
        logger.info(f"Starting retraining triggered by: {file_path}")
        retraining_start_time = time.time()
        
        try:
            # Load new data
            new_data = self.data_loader.load_dataset(str(Path(file_path).name))
            new_pairs = self.data_loader.combine_input_output_pairs(new_data)
            
            if not new_pairs:
                logger.warning(f"No valid pairs found in {file_path}")
                return
            
            # Perform continual learning
            results = self.trainer.continual_learn(
                Path(file_path),
                epochs=10 if self.incremental_mode else 50,
                mode="incremental" if self.incremental_mode else "full"
            )
            
            # Update statistics
            retraining_time = time.time() - retraining_start_time
            self.stats['retraining_count'] += 1
            self.stats['last_retraining'] = datetime.now().isoformat()
            self.stats['total_training_time'] += retraining_time
            self.stats['processed_files'] += 1
            
            # Log results
            logger.info(f"Retraining completed successfully:")
            logger.info(f"  Duration: {retraining_time:.2f} seconds")
            logger.info(f"  Final validation loss: {results.get('final_val_loss', 'N/A')}")
            logger.info(f"  Test loss: {results.get('test_loss', 'N/A')}")
            
            # Save statistics
            self._save_statistics()
            
            # Optional: Send notification
            self._send_notification(request, results, retraining_time)
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            self._save_error_log(file_path, str(e))
    
    def _save_statistics(self):
        """Save automation statistics to file."""
        stats_path = self.outputs_dir / "automation_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _send_notification(self, request: Dict[str, Any], results: Dict[str, Any], duration: float):
        """Send notification about completed retraining."""
        # This could be extended to send emails, Slack messages, etc.
        notification = {
            'timestamp': datetime.now().isoformat(),
            'file': request['file_path'],
            'duration': duration,
            'results': results
        }
        
        logger.info(f"Retraining notification: {notification}")
    
    def _save_error_log(self, file_path: str, error_message: str):
        """Save error log for failed retraining."""
        error_log_path = self.outputs_dir / "retraining_errors.log"
        
        with open(error_log_path, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {file_path}: {error_message}\n")
    
    def start_monitoring(self):
        """Start file system monitoring."""
        if self.is_running:
            logger.warning("Automation is already running")
            return
        
        self.is_running = True
        
        # Start retraining queue processor
        self.queue_thread = threading.Thread(target=self._process_retraining_queue, daemon=True)
        self.queue_thread.start()
        
        # Start file system observer
        self.observer = Observer()
        event_handler = RetrainingEventHandler(
            self.trainer,
            self.retraining_queue,
            self.processed_files
        )
        
        self.observer.schedule(event_handler, str(self.data_dir), recursive=WATCH_RECURSIVELY)
        self.observer.start()
        
        logger.info(f"Started monitoring directory: {self.data_dir}")
        logger.info("Automation is now active. Press Ctrl+C to stop.")
    
    def stop_monitoring(self):
        """Stop file system monitoring."""
        if not self.is_running:
            logger.warning("Automation is not running")
            return
        
        self.is_running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        # Wait for queue to empty
        self.retraining_queue.join()
        
        logger.info("Automation stopped")
    
    def trigger_manual_retraining(self, file_pattern: str = "*"):
        """
        Manually trigger retraining on all files matching pattern.
        
        Args:
            file_pattern: Glob pattern for files to retrain on
        """
        logger.info(f"Triggering manual retraining with pattern: {file_pattern}")
        
        # Load all data matching pattern
        dataset = self.data_loader.load_dataset(file_pattern)
        pairs = self.data_loader.combine_input_output_pairs(dataset)
        
        if not pairs:
            logger.warning("No valid data found for retraining")
            return
        
        # Perform full retraining
        X, y = self.data_loader.prepare_training_data(pairs)
        X_train, X_val, y_train, y_val = self.data_loader.split_data(X, y)[:4]
        
        results = self.trainer.train(X_train, y_train, X_val, y_val, epochs=100)
        
        logger.info(f"Manual retraining completed: {results}")
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        return {
            'is_running': self.is_running,
            'data_directory': str(self.data_dir),
            'models_directory': str(self.models_dir),
            'incremental_mode': self.incremental_mode,
            'queue_size': self.retraining_queue.qsize(),
            'processed_files_count': len(self.processed_files),
            'statistics': self.stats
        }

class ScheduledRetrainer:
    """
    Scheduled retraining using cron-like functionality.
    """
    
    def __init__(self, automation_manager: AutomationManager, interval_hours: int = 24):
        """
        Initialize scheduled retrainer.
        
        Args:
            automation_manager: AutomationManager instance
            interval_hours: Hours between scheduled retraining
        """
        self.automation_manager = automation_manager
        self.interval_hours = interval_hours
        self.is_running = False
        self.scheduler_thread = None
        
        logger.info(f"Initialized ScheduledRetrainer with {interval_hours}h interval")
    
    def start_scheduler(self):
        """Start the scheduled retraining."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Scheduled retraining started")
    
    def stop_scheduler(self):
        """Stop the scheduled retraining."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        logger.info("Scheduled retraining stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Wait for the specified interval
                time.sleep(self.interval_hours * 3600)
                
                if self.is_running:
                    logger.info("Triggering scheduled retraining")
                    self.automation_manager.trigger_manual_retraining()
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal")
    global automation_manager
    if automation_manager:
        automation_manager.stop_monitoring()
    sys.exit(0)

def main():
    """Main function for command-line automation."""
    parser = argparse.ArgumentParser(description="Automation for continual learning")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                       help="Directory to monitor for new data")
    parser.add_argument("--models_dir", type=str, default=str(MODELS_DIR),
                       help="Directory for model storage")
    parser.add_argument("--outputs_dir", type=str, default=str(OUTPUTS_DIR),
                       help="Directory for output files")
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental learning mode")
    parser.add_argument("--manual_retrain", action="store_true",
                       help="Trigger manual retraining and exit")
    parser.add_argument("--schedule", type=int, default=None,
                       help="Enable scheduled retraining with interval in hours")
    
    args = parser.parse_args()
    
    # Create automation manager
    global automation_manager
    automation_manager = AutomationManager(
        data_dir=Path(args.data_dir),
        models_dir=Path(args.models_dir),
        outputs_dir=Path(args.outputs_dir),
        incremental_mode=args.incremental
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.manual_retrain:
        # Perform manual retraining
        results = automation_manager.trigger_manual_retraining()
        print(f"Manual retraining completed: {results}")
        return
    
    # Start monitoring
    automation_manager.start_monitoring()
    
    # Start scheduled retraining if requested
    if args.schedule:
        scheduler = ScheduledRetrainer(automation_manager, args.schedule)
        scheduler.start_scheduler()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        automation_manager.stop_monitoring()
        if args.schedule:
            scheduler.stop_scheduler()

if __name__ == "__main__":
    main()
