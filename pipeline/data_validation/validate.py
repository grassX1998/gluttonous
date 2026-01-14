"""
数据验证模块 - Phase 2: Data Validation

负责验证清洗后的数据质量，确保数据可用于训练
"""

import sys
from pathlib import Path
import json

import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    CLEANED_DATA_DIR, FEATURE_DATA_DIR, VALIDATION_CONFIG
)
from pipeline.shared.utils import (
    setup_logger, timer, check_dataframe_quality
)


logger = setup_logger("data_validation")


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.config = VALIDATION_CONFIG
        self.validation_report = {
            "timestamp": None,
            "symbols_checked": 0,
            "symbols_passed": 0,
            "issues": [],
            "statistics": {}
        }
    
    @timer
    def validate_cleaned_data(self) -> dict:
        """验证清洗后的数据"""
        logger.info("Validating cleaned data...")
        
        cleaned_files = list(CLEANED_DATA_DIR.glob("*.parquet"))
        
        if len(cleaned_files) == 0:
            logger.error("No cleaned data found!")
            return {"error": "No cleaned data"}
        
        logger.info(f"Found {len(cleaned_files)} cleaned files")
        
        issues = []
        passed_count = 0
        
        for file_path in cleaned_files:
            symbol = file_path.stem
            
            try:
                df = pl.read_parquet(file_path)
                
                # 检查数据质量
                quality_report = check_dataframe_quality(df, symbol)
                
                # 检查缺失值
                if quality_report["missing"]:
                    issues.append({
                        "symbol": symbol,
                        "type": "missing_values",
                        "details": quality_report["missing"]
                    })
                
                # 检查数据量
                if df.height < 1000:
                    issues.append({
                        "symbol": symbol,
                        "type": "insufficient_data",
                        "details": f"Only {df.height} records"
                    })
                
                # 检查价格异常
                if "close" in df.columns:
                    close_prices = df["close"].to_numpy()
                    if np.any(close_prices <= 0):
                        issues.append({
                            "symbol": symbol,
                            "type": "invalid_price",
                            "details": "Non-positive prices found"
                        })
                
                # 如果没有问题，算作通过
                if not any(i["symbol"] == symbol for i in issues):
                    passed_count += 1
                
            except Exception as e:
                issues.append({
                    "symbol": symbol,
                    "type": "read_error",
                    "details": str(e)
                })
        
        self.validation_report["symbols_checked"] = len(cleaned_files)
        self.validation_report["symbols_passed"] = passed_count
        self.validation_report["issues"] = issues
        
        logger.info(f"Validation complete: {passed_count}/{len(cleaned_files)} passed")
        
        if issues:
            logger.warning(f"Found {len(issues)} issues")
            for issue in issues[:10]:  # 显示前10个
                logger.warning(f"  {issue['symbol']}: {issue['type']} - {issue['details']}")
        
        return self.validation_report
    
    @timer
    def validate_feature_data(self) -> dict:
        """验证特征数据"""
        logger.info("Validating feature data...")
        
        feature_files = list(FEATURE_DATA_DIR.glob("*.parquet"))
        
        if len(feature_files) == 0:
            logger.warning("No feature data found")
            return {"warning": "No feature data"}
        
        logger.info(f"Found {len(feature_files)} feature files")
        
        issues = []
        
        for file_path in feature_files:
            symbol = file_path.stem
            
            try:
                df = pl.read_parquet(file_path)
                
                # 检查是否包含NaN或Inf
                for col in df.columns:
                    if df[col].dtype in [pl.Float32, pl.Float64]:
                        values = df[col].to_numpy()
                        if np.any(np.isnan(values)):
                            issues.append({
                                "symbol": symbol,
                                "type": "nan_in_features",
                                "column": col
                            })
                        if np.any(np.isinf(values)):
                            issues.append({
                                "symbol": symbol,
                                "type": "inf_in_features",
                                "column": col
                            })
                
            except Exception as e:
                issues.append({
                    "symbol": symbol,
                    "type": "read_error",
                    "details": str(e)
                })
        
        logger.info(f"Feature validation: {len(feature_files) - len(issues)}/{len(feature_files)} passed")
        
        return {"checked": len(feature_files), "issues": issues}
    
    @timer
    def check_data_distribution(self) -> dict:
        """检查数据分布"""
        logger.info("Analyzing data distribution...")
        
        cleaned_files = list(CLEANED_DATA_DIR.glob("*.parquet"))[:100]  # 采样100个
        
        all_closes = []
        all_volumes = []
        
        for file_path in cleaned_files:
            try:
                df = pl.read_parquet(file_path)
                all_closes.extend(df["close"].to_list())
                all_volumes.extend(df["volume"].to_list())
            except:
                continue
        
        distribution = {
            "price": {
                "mean": float(np.mean(all_closes)),
                "std": float(np.std(all_closes)),
                "min": float(np.min(all_closes)),
                "max": float(np.max(all_closes)),
                "median": float(np.median(all_closes)),
            },
            "volume": {
                "mean": float(np.mean(all_volumes)),
                "std": float(np.std(all_volumes)),
                "min": float(np.min(all_volumes)),
                "max": float(np.max(all_volumes)),
                "median": float(np.median(all_volumes)),
            }
        }
        
        self.validation_report["statistics"]["distribution"] = distribution
        
        logger.info(f"Price range: [{distribution['price']['min']:.2f}, "
                   f"{distribution['price']['max']:.2f}], "
                   f"mean={distribution['price']['mean']:.2f}")
        
        return distribution
    
    @timer
    def run(self, save_report: bool = True):
        """执行完整的验证流程"""
        logger.info("="*60)
        logger.info("Starting Data Validation Pipeline")
        logger.info("="*60)
        
        from datetime import datetime
        self.validation_report["timestamp"] = datetime.now().isoformat()
        
        # 1. 验证清洗数据
        self.validate_cleaned_data()
        
        # 2. 检查数据分布
        self.check_data_distribution()
        
        # 3. 验证特征数据（如果存在）
        feature_result = self.validate_feature_data()
        self.validation_report["feature_validation"] = feature_result
        
        # 4. 保存报告
        if save_report and self.config["generate_report"]:
            report_path = CLEANED_DATA_DIR.parent / "validation_report.json"
            with open(report_path, "w") as f:
                json.dump(self.validation_report, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")
        
        # 5. 输出总结
        self._print_summary()
        
        return self.validation_report
    
    def _print_summary(self):
        """打印验证总结"""
        logger.info("\n" + "="*60)
        logger.info("Data Validation Summary")
        logger.info("="*60)
        logger.info(f"Symbols checked: {self.validation_report['symbols_checked']}")
        logger.info(f"Symbols passed: {self.validation_report['symbols_passed']}")
        
        if self.validation_report['symbols_checked'] > 0:
            pass_rate = (self.validation_report['symbols_passed'] / 
                        self.validation_report['symbols_checked'] * 100)
            logger.info(f"Pass rate: {pass_rate:.1f}%")
        
        logger.info(f"Issues found: {len(self.validation_report['issues'])}")
        logger.info("="*60)


def main():
    """主函数"""
    validator = DataValidator()
    validator.run()


if __name__ == "__main__":
    main()
