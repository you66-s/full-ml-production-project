from feast import Field, FeatureView
from feast.types import Float32, Int64, Bool, String
from entities import users
from data_sources import (subscription_profile_src, usage_agg_30d_src, payments_agg_90d_src, support_agg_90d_src)

"""
This file defines wich features collection to use for model training and serving with associated entites data schema and data source
"""

subscription_profile_fv = FeatureView(
    name="subscription_profile_fv", 
    entities=[users],
    ttl=None,
    schema=[
        Field(name="months_active", dtype=Int64),
        Field(name="monthly_fee", dtype=Float32),
        Field(name="paperless_billing", dtype=Bool),
        Field(name="plan_stream_tv", dtype=Bool),
        Field(name="plan_stream_movies", dtype=Bool),
        Field(name="net_service", dtype=String),
    ],
    source=subscription_profile_src,
    online=True,
    tags={"owner": "mlops-course"},
)

usage_agg_30d_fv = FeatureView(
    name="usage_agg_30d_fv", 
    entities=[users],
    ttl=None,
    schema=[
        Field(name="watch_hours_30d", dtype=Float32),
        Field(name="avg_session_mins_7d", dtype=Float32),
        Field(name="unique_devices_30d", dtype=Int64),
        Field(name="skips_7d", dtype=Int64),
        Field(name="rebuffer_events_7d", dtype=Int64),
    ],
    source=usage_agg_30d_src,
    online=True,
    tags={"owner": "mlops-course"},
)

payments_agg_90d_fv = FeatureView(
    name="payments_agg_90d_fv", 
    entities=[users],
    ttl=None,
    schema=[
        Field(name="failed_payments_90d", dtype=Int64)
    ],
    source=payments_agg_90d_src,
    online=True,
    tags={"owner": "mlops-course"},
)

support_agg_90d_fv = FeatureView(
    name="support_agg_90d_fv", 
    entities=[users],
    ttl=None,
    schema=[
        Field(name="support_tickets_90d", dtype=Int64),
        Field(name="ticket_avg_resolution_hrs_90d", dtype=Float32)
    ],
    source=support_agg_90d_src,
    online=True,
    tags={"owner": "mlops-course"},
)


"""
feast apply: to take configuration and registers them in Feast.

"""