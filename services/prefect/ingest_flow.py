from core.config import get_settings
from core.database import engine
from sqlalchemy import text
from prefect import task, flow
import pandas as pd
import great_expectations as gx


settings = get_settings()

def process_users_table(csv_path: str):
    dataset = pd.read_csv(csv_path)
    dataset["signup_date"] = pd.to_datetime(dataset["signup_date"], errors="coerce")
    return dataset

def process_usage_table(csv_path: str):
    dataset = pd.read_csv(csv_path)
    return dataset

def process_payments_table(csv_path: str):
    return pd.read_csv(csv_path)

def process_support_table(csv_path: str):
    return pd.read_csv(csv_path)

def process_subscriptions_table(csv_path: str):
    dataset = pd.read_csv(csv_path)
    dataset["plan_stream_tv"] = dataset['plan_stream_tv'].astype("bool")
    dataset["plan_stream_movies"] = dataset['plan_stream_movies'].astype("bool")
    return dataset

def process_labels_table(csv_path: str):
    return pd.read_csv(csv_path)


@task
def upsert_csv(table: str, csv_path: str, pk_cols: list[str]):
    dataframe = pd.read_csv(csv_path)
    
    # conversion et nettoyage des données selon la table
    if table == "users":
        dataframe = process_users_table(csv_path=csv_path)
    elif table == "usage_agg_30d":
        dataframe = process_usage_table(csv_path=csv_path)
    elif table == "support_agg_90d":
        dataframe = process_support_table(csv_path=csv_path)
    elif table == "subscriptions":
        dataframe = process_subscriptions_table(csv_path=csv_path)
    elif table == "payments_agg_90d":
        dataframe = process_payments_table(csv_path=csv_path)
    elif table == "labels":
        dataframe = process_labels_table(csv_path=csv_path)
        
    with engine.begin() as conn:
        tmp = f"tmp_{table}"
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {tmp}")
        dataframe.head(0).to_sql(name=tmp, con=conn, if_exists="replace", index=False)
        dataframe.to_sql(name=tmp, con=conn, if_exists="append", index=False)
        cols = list(dataframe.columns)
        col_list = ", ".join(cols)
        pk = ", ".join(pk_cols)
        
        updates = ", ".join(
            [
                f"{col} = EXCLUDED.{col}" for col in cols if col not in pk_cols
            ]
        )
        
        sql = text(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {tmp} ON CONFLICT ({pk}) DO UPDATE SET {updates}")
        conn.execute(sql)
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {tmp}")
    
    return f"upserted {len(dataframe)} rows into {table}"

@task(name="Data Validation based on expectations pre-defined")
def validate_upserted_data(table: str):
    with engine.begin() as conn:
        dataframe = pd.read_sql(text(f"SELECT * FROM {table}"), con=conn)
        context = gx.get_context()
        data_source_name  = "pandas_postgres_dataframe"
        data_source = context.data_sources.add_pandas(data_source_name)
        
        # validation rules
        if table == "users":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=["user_id","signup_date","user_gender","user_is_senior","has_family","has_dependents"])
            )
            expectations.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
            )
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        elif table == "usage_agg_30d":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=[
                    "user_id","watch_hours_30d","avg_session_mins_7d",
                    "unique_devices_30d","skips_7d","rebuffer_events_7d"])
            )
            expectations.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
            )
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        elif table == "support_agg_90d":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=[
                    "user_id", "watch_hours_30d", "avg_session_mins_7d", "unique_devices_30d",
                    "skips_7d", "rebuffer_events_7d"
                ])
            )
            expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="watch_hours_30d", min_value=0, max_value=721))
            expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="avg_session_mins_7d", min_value=0))
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        elif table == "subscriptions":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=[
                    "user_id", "months_active", "plan_stream_tv", "plan_stream_movies",
                    "contract_type", "paperless_billing", "monthly_fee", "total_paid",
                    "net_service", "add_on_security", "add_on_backup",
                    "add_on_device_protect", "add_on_support"
                ])
            )
            expectations.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
            expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="months_active", min_value=0))
            expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="monthly_fee", min_value=0))
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        elif table == "payments_agg_90d":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=[
                    "user_id", "failed_payments_90d"
                ])
            )
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        elif table == "labels":
            data_asset_name = f"{table}_data_asset"
            data_asset = data_source.add_dataframe_asset(name=data_asset_name)
            batch_definition_name = f"{table}_batch_definition"
            batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
            batch_parameters = {"dataframe": dataframe}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)
            expectations = gx.ExpectationSuite(name=f"{table}_expectations_suite")
            expectations.add_expectation(
                gx.expectations.ExpectTableColumnsToMatchSet(column_set=[
                    "user_id", "failed_payments_90d"
                ])
            )
            validation_results = batch.validate(expectations)
            if not validation_results.success:
                raise ValueError(f"Validation failed for {table}: {validation_results}")
            return f"GE passed for {table}"
        
@task(name="les tables de snapshots")
def build_month_snapshot(as_of: str):
    """
    Crée (si besoin) les tables de snapshots et insère les données
    pour la date as_of donnée. Utilise une stratégie idempotente
    (ON CONFLICT DO NOTHING).
    """
    ddl = """
        CREATE TABLE IF NOT EXISTS subscriptions_profile_snapshots (
        user_id TEXT,
        as_of DATE,
        months_active INT,
        monthly_fee NUMERIC,
        paperless_billing BOOLEAN,
        plan_stream_tv BOOLEAN,
        plan_stream_movies BOOLEAN,
        net_service TEXT,
        PRIMARY KEY (user_id, as_of)
        );

        CREATE TABLE IF NOT EXISTS usage_agg_30d_snapshots (
        user_id TEXT,
        as_of DATE,
        watch_hours_30d NUMERIC,
        avg_session_mins_7d NUMERIC,
        unique_devices_30d INT,
        skips_7d INT,
        rebuffer_events_7d INT,
        PRIMARY KEY (user_id, as_of)
        );

        CREATE TABLE IF NOT EXISTS payments_agg_90d_snapshots (
        user_id TEXT,
        as_of DATE,
        failed_payments_90d INT,
        PRIMARY KEY (user_id, as_of)
        );

        CREATE TABLE IF NOT EXISTS support_agg_90d_snapshots (
        user_id TEXT,
        as_of DATE,
        support_tickets_90d INT,
        ticket_avg_resolution_hrs_90d NUMERIC,
        PRIMARY KEY (user_id, as_of)
        );
    """
    sqls = [
        f"""
        INSERT INTO subscriptions_profile_snapshots
        (user_id, as_of, months_active, monthly_fee, paperless_billing,
         plan_stream_tv, plan_stream_movies, net_service)
        SELECT user_id, DATE '{as_of}', months_active, monthly_fee, paperless_billing,
               plan_stream_tv, plan_stream_movies, net_service
        FROM subscriptions
        ON CONFLICT (user_id, as_of) DO NOTHING;
        """,
        f"""
        INSERT INTO usage_agg_30d_snapshots
        (user_id, as_of, watch_hours_30d, avg_session_mins_7d,
         unique_devices_30d, skips_7d, rebuffer_events_7d)
        SELECT user_id, DATE '{as_of}', watch_hours_30d, avg_session_mins_7d,
               unique_devices_30d, skips_7d, rebuffer_events_7d
        FROM usage_agg_30d
        ON CONFLICT (user_id, as_of) DO NOTHING;
        """,
        f"""
        INSERT INTO payments_agg_90d_snapshots
        (user_id, as_of, failed_payments_90d)
        SELECT
            user_id, DATE '{as_of}', failed_payments_90d
        FROM payments_agg_90d
        ON CONFLICT (user_id, as_of) DO NOTHING;
        """,
        f"""
        INSERT INTO support_agg_90d_snapshots
        (user_id, as_of, support_tickets_90d, ticket_avg_resolution_hrs_90d)
        SELECT user_id, DATE '{as_of}', support_tickets_90d, ticket_avg_resolution_hrs_90d
        FROM support_agg_90d
        ON CONFLICT (user_id, as_of) DO NOTHING;
        """
    ]
    
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)
        for sql in sqls:
            conn.exec_driver_sql(sql)
    
    return f"snapshots stamped for {as_of}"
@flow(name="ingest_month")
def ingest_month_flow(seed_dir: str = "/data/seeds/month_001", as_of: str = "2022-02-28"):
    upsert_csv("users",            f"{seed_dir}/users.csv",            ["user_id"])
    upsert_csv("subscriptions",    f"{seed_dir}/subscriptions.csv",    ["user_id"])
    upsert_csv("usage_agg_30d",    f"{seed_dir}/usage_agg_30d.csv",    ["user_id"])
    upsert_csv("payments_agg_90d", f"{seed_dir}/payments_agg_90d.csv", ["user_id"])
    upsert_csv("support_agg_90d",  f"{seed_dir}/support_agg_90d.csv",  ["user_id"])
    upsert_csv("labels",           f"{seed_dir}/labels.csv",           ["user_id"])
    
    # Validation GE (garde-fou avant les snapshots)
    validate_upserted_data("users")
    validate_upserted_data("subscriptions")
    validate_upserted_data("usage_agg_30d")
    
    # Snapshots temporels
    build_month_snapshot(as_of=as_of)
    return f"Ingestion et validation terminée pour {as_of}"

if __name__ == "__main__":
    ingest_month_flow()