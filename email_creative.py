"""Escaped email HTML and a downloadable MIME email with an embedded hero image."""
import base64
from email.message import EmailMessage
from email.policy import SMTP
from html import escape
from urllib.parse import urlparse


def validate_booking_url(url):
    if not url:
        return ''
    parsed = urlparse(url)
    if parsed.scheme not in ('https', 'http') or not parsed.netloc:
        raise ValueError('Enter a full http:// or https:// booking URL, or leave it blank.')
    return url


def render_email(hotel, features, copy, image=None, booking_url='', image_src=None):
    booking_url = validate_booking_url(booking_url)
    e = lambda value: escape(str(value), quote=True)
    hero = ''
    if image:
        src = image_src or 'data:image/png;base64,' + base64.b64encode(image).decode()
        hero = f'<img src="{e(src)}" alt="Illustration inspired by guest-praised hotel features" width="600" style="width:100%;display:block;">'
    cards = ''.join(f'<tr><td style="padding:14px 32px;border-top:1px solid #34423f;">'
                    f'<h3 style="color:#9ee3c1;margin:0 0 8px;">{e(f["title"])}</h3>'
                    f'<p style="margin:0;line-height:1.6;">{e(f["description"])}</p></td></tr>' for f in features)
    cta = (f'<a href="{e(booking_url)}" style="display:inline-block;background:#9ee3c1;color:#10241b;'
           f'padding:16px 24px;border-radius:5px;text-decoration:none;">{e(copy["cta"])}</a>') if booking_url else '<p>Contact the hotel to enquire about this offer.</p>'
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0;background:#0e1418;color:#eef4f1;font-family:Arial,sans-serif;">
<div style="display:none;max-height:0;overflow:hidden;">{e(copy['preheader'])}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0"><tr><td align="center" style="padding:24px 8px;">
<table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:#182229;">
<tr><td style="padding:28px 32px;color:#9ee3c1;">{e(hotel['name'])} · {e(hotel['city'])}</td></tr>
<tr><td>{hero}</td></tr><tr><td style="padding:32px;">
<p style="color:#9ee3c1;letter-spacing:2px;">2 days, 3 nights</p>
<h1 style="font-size:34px;line-height:1.15;">{e(copy['headline'])}</h1>
<p style="line-height:1.7;">{e(copy['introduction'])}</p><h2 style="font-size:20px;">Inspired by what guests love</h2></td></tr>
{cards}<tr><td style="padding:32px;"><p style="line-height:1.7;">{e(copy['closing'])}</p>{cta}
<p style="font-size:12px;color:#aabbb4;margin-top:28px;">Illustration inspired by historical guest reviews. Contact the hotel for availability and offer details.</p>
</td></tr></table></td></tr></table></body></html>'''


def export_email(hotel, features, copy, image, booking_url=''):
    message = EmailMessage(policy=SMTP)
    message['Subject'] = ' '.join(copy['subject'].splitlines())
    message['X-Unsent'] = '1'
    body = f"{hotel['name']} — 2 days, 3 nights\n\n{copy['introduction']}\n\n"
    body += '\n'.join(f"{f['title']}: {f['description']}" for f in features)
    body += f"\n\n{copy['closing']}\n{booking_url or 'Contact the hotel to enquire.'}"
    message.set_content(body)
    message.add_alternative(render_email(hotel, features, copy, image, booking_url, 'cid:hotel-hero'), subtype='html')
    if image:
        message.get_payload()[-1].add_related(image, maintype='image', subtype='png', cid='<hotel-hero>', filename='hotel-promotion.png')
    return message.as_bytes()
