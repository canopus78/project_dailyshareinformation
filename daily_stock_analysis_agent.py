import os
from datetime import datetime

# --- 이메일 모듈을 함수 안에서 직접 임포트 ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# -----------------------------------------

def simple_email_test():
    print("--- Starting the simplified email test. ---")

    gmail_user = os.environ.get('GMAIL_USERNAME')
    gmail_password = os.environ.get('GMAIL_PASSWORD')
    recipient_email = os.environ.get('RECIPIENT_EMAIL')

    if not all([gmail_user, gmail_password, recipient_email]):
        print("!!! CRITICAL ERROR: Email environment variables (secrets) are NOT set.")
        return

    print("Secrets loaded successfully.")

    # 이메일 메시지 생성
    msg = MIMEMultipart()
    current_date = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"✅ [Test Success] GitHub Actions Email Test - {current_date}"
    msg['From'] = gmail_user
    msg['To'] = recipient_email

    html_body = "<h1>GitHub Actions Email Test</h1><p>This is a test email to confirm the mailing function is working correctly.</p>"
    
    print("About to create MIMEText object...")
    # 문제가 되는 부분을 직접 테스트
    text_part = MIMEText(html_body, 'html')
    print("MIMEText object created successfully.")

    msg.attach(text_part)
    print("Message object assembled.")

    # Gmail SMTP 서버를 통해 이메일 발송
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_password)
        server.send_message(msg)
        server.quit()
        print('✅✅✅ Email sent successfully! The problem is likely solved.')
    except Exception as e:
        print(f'❌❌❌ An error occurred while sending the email: {e}')
        raise

if __name__ == "__main__":
    try:
        simple_email_test()
    except Exception as e:
        print(f"!!! A critical error occurred during script execution: {e}")
        exit(1)