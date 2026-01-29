"""
指标和结果记录模块
"""

from .result_recorder import ResultRecorder
from .report_generator import BacktestReportGenerator, generate_report_from_json

__all__ = [
    'ResultRecorder',
    'BacktestReportGenerator',
    'generate_report_from_json',
]
