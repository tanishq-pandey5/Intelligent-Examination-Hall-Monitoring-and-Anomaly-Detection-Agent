"""
Intelligent Examination Hall Monitoring and Anomaly Detection Agent - Core Package.
"""
from src.video_input import VideoInputHandler, VideoMetadata
from src.frame_sampler import FrameSampler, VideoFrame
from src.detector import ExamHallDetector, DetectedObject
from src.tracker import PersonTracker, TrackedPerson
from src.seat_mapper import SeatMapper, SeatOccupancyRecord
from src.behavior import BehaviorAnalyzer, BehaviorObservation
from src.agent import AgentState, ReviewState, PersonAgentMemory
from src.reasoning import RuleBasedReasoningEngine, RuleMatchResult
from src.risk_engine import RiskEngine, RiskBreakdown
from src.explanations import AlertExplainer, ExplainableAlert
from src.evidence import EvidenceManager
from src.database import DatabaseManager
from src.report_generator import ReportGenerator
from src.ml_anomaly_model import ExamAnomalyDetector
from src.dataset_builder import ExamDatasetBuilder
from src.trainer import ExamModelTrainer

__all__ = [
    "VideoInputHandler",
    "VideoMetadata",
    "FrameSampler",
    "VideoFrame",
    "ExamHallDetector",
    "DetectedObject",
    "PersonTracker",
    "TrackedPerson",
    "SeatMapper",
    "SeatOccupancyRecord",
    "BehaviorAnalyzer",
    "BehaviorObservation",
    "AgentState",
    "ReviewState",
    "PersonAgentMemory",
    "RuleBasedReasoningEngine",
    "RuleMatchResult",
    "RiskEngine",
    "RiskBreakdown",
    "AlertExplainer",
    "ExplainableAlert",
    "EvidenceManager",
    "DatabaseManager",
    "ReportGenerator",
    "ExamAnomalyDetector",
    "ExamDatasetBuilder",
    "ExamModelTrainer",
]

