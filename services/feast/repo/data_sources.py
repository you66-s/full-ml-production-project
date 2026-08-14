from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

subscription_profile_src = PostgreSQLSource(
    name="subscription_profile_src",
    query="SELECT user_id, as_of, months_active, monthly_fee, paperless_billing, plan_stream_tv, plan_stream_movies, net_service FROM subscriptions_profile_snapshots",
    timestamp_field="as_of"
)

usage_agg_30d_src = PostgreSQLSource(
    name="usage_agg_30d_src",
    query="SELECT user_id, as_of, watch_hours_30d, avg_session_mins_7d, unique_devices_30d, skips_7d, rebuffer_events_7d FROM usage_agg_30d_snapshots",
    timestamp_field="as_of"
)

payments_agg_90d_src = PostgreSQLSource(
    name="payments_agg_90d_src",
    query="SELECT user_id, as_of, failed_payments_90d FROM payments_agg_90d_snapshots",
    timestamp_field="as_of"
)

support_agg_90d_src = PostgreSQLSource(
    name="support_agg_90d_src",
    query="SELECT user_id, as_of, support_tickets_90d, ticket_avg_resolution_hrs_90d FROM support_agg_90d_snapshots",
    timestamp_field="as_of"
)