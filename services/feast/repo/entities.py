from feast import Entity
from feast.value_type import ValueType

users = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.STRING
)
