from django import template
from django.utils import timezone
from django.utils.timesince import timesince

register = template.Library()

@register.filter
def humanize_date(value):
    now = timezone.now()
    if not value:
        return ""
    if value.tzinfo is None:
        value = timezone.make_aware(value, timezone.get_current_timezone())
    diff = now - value

    if diff.days == 0:
        if diff.seconds < 60:
            return "just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60} minutes ago"
        else:
            return f"{diff.seconds // 3600} hours ago"
    elif diff.days == 1:
        return "yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    else:
        return value.strftime('%B %d, %Y')
