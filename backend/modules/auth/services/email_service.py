import logging

logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    def send_verification_email(email: str, token: str) -> None:
        verification_link = f"http://localhost:3000/verify-email?token={token}"
        logger.info(f"发送验证邮件到 {email}。链接: {verification_link}")
        print(f"--- EMAIL TO {email} ---")
        print(f"Please verify your email: {verification_link}")
        print("------------------------")

    @staticmethod
    def send_reset_password_email(email: str, token: str) -> None:
        reset_link = f"http://localhost:3000/reset-password?token={token}"
        logger.info(f"发送密码重置邮件到 {email}。链接: {reset_link}")
        print(f"--- EMAIL TO {email} ---")
        print(f"Reset your password here: {reset_link}")
        print("------------------------")

    @staticmethod
    def send_account_deleted_email(email: str, name: str) -> None:
        logger.info(f"发送账号注销邮件到 {email}。")
        print(f"--- EMAIL TO {email} ---")
        print(f"Hello {name}, your Paper2Code account has been successfully deleted.")
        print("------------------------")
