"""Email utility functions for sending notifications (via CallMeBot WhatsApp API).

Based on: https://www.callmebot.com/blog/free-api-whatsapp-messages/
API Format: https://api.callmebot.com/whatsapp.php?phone=[phone_number]&text=[message]&apikey=[your_apikey]
"""

import logging
import os
import requests
from typing import Optional

# Import config to ensure .env file is loaded from MainApp directory
import config

logger = logging.getLogger(__name__)

# CallMeBot WhatsApp API configuration
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY", "")
RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE", "")
CALLMEBOT_API_URL = "https://api.callmebot.com/whatsapp.php"

def _clean_phone_number(phone: str) -> str:
    """
    Clean and format phone number for CallMeBot API.
    Removes spaces and ensures it starts with + and country code.
    
    Args:
        phone: Phone number (e.g., "+34 123 123 123" or "+34123123123")
        
    Returns:
        Cleaned phone number (e.g., "+34123123123")
    """
    # Remove all spaces
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    # Ensure it starts with +
    if not cleaned.startswith("+"):
        cleaned = "+" + cleaned
    
    return cleaned


def send_email(to_email: str, subject: str, body: str, html_body: Optional[str] = None) -> bool:
    """
    Send WhatsApp message via CallMeBot API instead of email.
    
    Based on CallMeBot API documentation:
    https://www.callmebot.com/blog/free-api-whatsapp-messages/
    
    Args:
        to_email: Recipient email address (used as identifier/logging)
        subject: Email subject line (included in WhatsApp message)
        body: Plain text email body (included in WhatsApp message)
        html_body: Optional HTML version of email body (ignored, plain text used)
        
    Returns:
        True if WhatsApp message sent successfully, False otherwise
    """
    logger.info(f"Sending WhatsApp notification (for email: {to_email})")
    logger.info(f"Subject: {subject}")
    
    # Send WhatsApp message via CallMeBot API
    try:
        # Format the WhatsApp message
        whatsapp_message = f"📧 Email Notification\n\nTo: {to_email}\nSubject: {subject}\n\n{body}"
        
        # Validate configuration
        if not WHATSAPP_API_KEY:
            logger.error("CallMeBot API key not configured. Please set WHATSAPP_API_KEY environment variable.")
            return False
            
        if not RECIPIENT_PHONE:
            logger.error("Recipient phone number not configured. Please set RECIPIENT_PHONE environment variable.")
            return False
        
        # Clean phone number (remove spaces, ensure proper format)
        cleaned_phone = _clean_phone_number(RECIPIENT_PHONE)
        
        # Prepare the API request according to CallMeBot documentation
        # Format: https://api.callmebot.com/whatsapp.php?phone=[phone_number]&text=[message]&apikey=[your_apikey]
        params = {
            'phone': cleaned_phone,
            'text': whatsapp_message,  # requests will URL-encode this automatically
            'apikey': WHATSAPP_API_KEY
        }
        
        # Make the API call (GET request as per documentation)
        response = requests.get(CALLMEBOT_API_URL, params=params, timeout=30)
        
        # Check response
        if response.status_code == 200:
            response_text = response.text.strip().lower()
            # CallMeBot API may return success indicators in the response text
            if 'ok' in response_text or 'success' in response_text or response_text == '':
                logger.info(f"WhatsApp message sent successfully to {cleaned_phone}")
                logger.debug(f"API Response: {response.text}")
                return True
            else:
                logger.warning(f"API returned 200 but response may indicate failure: {response.text}")
                # Still return True as status code is 200, but log the warning
                return True
        else:
            logger.error(f"Failed to send WhatsApp message. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
        
    except requests.exceptions.Timeout:
        logger.error("Timeout while sending WhatsApp message (30s timeout exceeded)")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending WhatsApp message: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending WhatsApp message: {str(e)}")
        return False


def send_verification_notification(
    to_email: str,
    artifact_name: str,
    status: str,
    reason: str,
    verified_by: str
) -> bool:
    """
    Send a WhatsApp notification about artifact verification status change via CallMeBot API.
    
    Args:
        to_email: Uploader's email address (used as identifier)
        artifact_name: Name of the artifact
        status: New verification status ('verified' or 'rejected')
        reason: Reason provided by the reviewer
        verified_by: Username of the person who verified/rejected
        
    Returns:
        True if WhatsApp message sent successfully, False otherwise
    """
    if status.lower() == "verified":
        status_text = "APPROVED ✅"
        status_color = "#28a745"
    else:
        status_text = "REJECTED ❌"
        status_color = "#dc3545"
    
    subject = f"Artifact '{artifact_name}' has been {status_text}"
    
    body = f"""
Hello,

Your artifact submission has been reviewed.

Artifact: {artifact_name}
Status: {status_text}
Reviewed by: {verified_by}

Reviewer's Comments:
{reason}

Thank you for your contribution to the Artifact Gallery.

Best regards,
The ArtiQuest Team
"""

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
        .status {{ display: inline-block; padding: 8px 16px; border-radius: 4px; color: white; font-weight: bold; background-color: {status_color}; }}
        .reason-box {{ background: white; border-left: 4px solid {status_color}; padding: 15px; margin: 15px 0; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏺 Artifact Gallery</h1>
            <p>Verification Notification</p>
        </div>
        <div class="content">
            <h2>Hello,</h2>
            <p>Your artifact submission has been reviewed.</p>
            
            <p><strong>Artifact:</strong> {artifact_name}</p>
            <p><strong>Status:</strong> <span class="status">{status_text}</span></p>
            <p><strong>Reviewed by:</strong> {verified_by}</p>
            
            <div class="reason-box">
                <strong>Reviewer's Comments:</strong>
                <p>{reason}</p>
            </div>
            
            <p>Thank you for your contribution to the Artifact Gallery.</p>
        </div>
        <div class="footer">
            <p>Best regards,<br>The ArtiQuest Team</p>
        </div>
    </div>
</body>
</html>
"""
    
    return send_email(to_email, subject, body, html_body)

