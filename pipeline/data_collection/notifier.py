"""
通知模块

支持通过邮件发送数据采集结果通知。
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

from pipeline.shared.config import EMAIL_CONFIG

logger = logging.getLogger(__name__)


def send_email(
    subject: str,
    body: str,
    recipient: Optional[str] = None,
) -> bool:
    """
    发送邮件

    Args:
        subject: 邮件主题
        body: 邮件正文（支持 HTML）
        recipient: 收件人邮箱（默认从配置读取）

    Returns:
        是否发送成功
    """
    if not EMAIL_CONFIG.get("enabled", False):
        logger.info("邮件通知已禁用")
        return False

    if not EMAIL_CONFIG.get("sender_password"):
        logger.warning("邮件密码未配置，跳过发送。请在 config.py 中配置 EMAIL_CONFIG['sender_password']")
        return False

    try:
        # 创建邮件
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_CONFIG["sender_email"]
        msg["To"] = recipient or EMAIL_CONFIG["recipient_email"]

        # 添加正文
        html_part = MIMEText(body, "html", "utf-8")
        msg.attach(html_part)

        # 连接 SMTP 服务器
        if EMAIL_CONFIG.get("use_tls", True):
            server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])

        # 登录并发送
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
        server.send_message(msg)
        server.quit()

        logger.info(f"邮件发送成功: {subject}")
        return True

    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


def send_collection_report(
    date: str,
    success_count: int,
    fail_count: int,
    status: str = "completed",
    error: Optional[str] = None,
) -> bool:
    """
    发送数据采集报告邮件

    Args:
        date: 采集日期
        success_count: 成功数
        fail_count: 失败数
        status: 状态
        error: 错误信息

    Returns:
        是否发送成功
    """
    # 确定状态图标和颜色
    if status == "completed" and fail_count == 0:
        status_icon = "✅"
        status_text = "成功"
        status_color = "#28a745"
    elif status == "completed_with_errors":
        status_icon = "⚠️"
        status_text = "部分失败"
        status_color = "#ffc107"
    else:
        status_icon = "❌"
        status_text = "失败"
        status_color = "#dc3545"

    subject = f"{status_icon} Gluttonous 数据采集报告 - {date}"

    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: {status_color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
            .content {{ background: #f9f9f9; padding: 20px; border-radius: 0 0 8px 8px; }}
            .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-item {{ text-align: center; }}
            .stat-value {{ font-size: 36px; font-weight: bold; color: {status_color}; }}
            .stat-label {{ font-size: 14px; color: #666; }}
            .error-box {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 4px; margin-top: 15px; }}
            .footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2 style="margin: 0;">{status_icon} 数据采集 {status_text}</h2>
                <p style="margin: 10px 0 0 0;">采集日期: {date}</p>
            </div>
            <div class="content">
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">{success_count}</div>
                        <div class="stat-label">成功</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{fail_count}</div>
                        <div class="stat-label">失败</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{success_count + fail_count}</div>
                        <div class="stat-label">总计</div>
                    </div>
                </div>
                {"<div class='error-box'><strong>错误信息:</strong><br>" + error + "</div>" if error else ""}
            </div>
            <div class="footer">
                <p>此邮件由 Gluttonous 量化系统自动发送</p>
                <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return send_email(subject, body)


def test_email() -> bool:
    """
    测试邮件发送

    Returns:
        是否成功
    """
    subject = "🧪 Gluttonous 邮件测试"
    body = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body>
        <h2>邮件配置测试成功！</h2>
        <p>如果您收到此邮件，说明邮件通知功能已正确配置。</p>
        <p>发送时间: {}</p>
    </body>
    </html>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    return send_email(subject, body)


if __name__ == "__main__":
    # 测试邮件发送
    logging.basicConfig(level=logging.INFO)
    test_email()
