import os
import ssl
import smtplib
from email.message import EmailMessage
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

class Emailer:
    def __init__(self, templates_dir: str = "templates"):
        self.sender = os.getenv("EMAIL_SENDER")
        self.password = os.getenv("SMTP_APP_PASSWORD")

        self.jinja_env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)

    def send_email(self, to: str, subject: str, template: str, variables: dict) -> str:
        try:
            try:
                tmpl = self.jinja_env.get_template(f"{template}.html")
                html_body = tmpl.render(**variables)
            except TemplateNotFound:
                return f"ERROR: Template '{template}' not found."
            except Exception as e:
                return f"ERROR: Failed to render template: {e}"

            msg = EmailMessage()
            msg["From"] = f"The PlanPerfect Team <{self.sender}>"
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content("This email requires an HTML-compatible email client.")
            msg.add_alternative(html_body, subtype="html")

            smtp_host = os.getenv("SMTP_HOST")
            smtp_port = int(os.getenv("SMTP_PORT"))

            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(self.sender, self.password)
                server.send_message(msg)

            return "SUCCESS: Email sent successfully."
        except Exception as e:
            return f"ERROR: Failed to send email. Error: {e}"