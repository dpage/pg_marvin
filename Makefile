EXTENSION = pg_marvin
DATA = pg_marvin--1.0.sql
PGFILEDESC = "Machine learning and analytics for PostgreSQL"

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
