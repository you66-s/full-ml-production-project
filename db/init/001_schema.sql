CREATE TABLE IF NOT EXISTS users(
    user_id TEXT PRIMARY KEY,
    signup_date DATE,
    user_gender TEXT,
    user_is_senior BOOLEAN,
    has_family BOOLEAN,
    has_dependents BOOLEAN
);

CREATE TABLE IF NOT EXISTS subscriptions (
    user_id TEXT REFERENCES users(user_id),
    months_active INT,
    plan_stream_tv BOOLEAN,
    plan_stream_movies BOOLEAN,
    contract_type TEXT,
    paperless_billing BOOLEAN,
    monthly_fee NUMERIC,
    total_paid NUMERIC,
    net_service TEXT,
    -- hidden at start (left NULL)
    add_on_security BOOLEAN,
    add_on_backup BOOLEAN,
    add_on_device_protect BOOLEAN,
    add_on_support BOOLEAN,
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS usage_agg_30d (
    user_id TEXT REFERENCES users(user_id),
    watch_hours_30d NUMERIC,
    avg_session_mins_7d NUMERIC,
    unique_devices_30d INT,
    skips_7d INT,
    rebuffer_events_7d INT,
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS payments_agg_90d (
    user_id TEXT REFERENCES users(user_id),
    failed_payments_90d INT,
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS support_agg_90d (
    user_id TEXT REFERENCES users(user_id),
    support_tickets_90d INT,
    ticket_avg_resolution_hrs_90d NUMERIC,
    PRIMARY KEY (user_id)
);

CREATE TABLE IF NOT EXISTS labels (
    user_id TEXT REFERENCES users(user_id),
    churn_label BOOLEAN,
    PRIMARY KEY (user_id)
);