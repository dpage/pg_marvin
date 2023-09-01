--
-- Create the virtual environment
--
CREATE OR REPLACE FUNCTION @extschema@.create_venv()
    RETURNS boolean
    LANGUAGE sql
AS $BODY$
    SELECT plpy_venv.create_venv('marvin');
    SELECT plpy_venv.activate_venv('marvin');
    SELECT plpy_venv.pip_install('{transformers[torch]}');
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.create_venv() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.create_venv() TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.create_venv() IS 'Create a virtual environment for running pg_marvin.';


--
-- Run a pipeline task
--
CREATE OR REPLACE FUNCTION run_pipeline(data IN text[], task IN text, model IN text DEFAULT NULL)
    RETURNS json
    LANGUAGE plpython3u
AS $BODY$
from transformers import pipeline
import json

plpy.execute("SELECT plpy_venv.activate_venv('marvin');")
classifier = pipeline(task, model=model)

return json.dumps(classifier(data))
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.run_pipeline(text[], text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.run_pipeline(text[], text, text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.run_pipeline(text[], text, text) IS 'Run a pipeline task.';


--
-- Analyse the sentiment of an array of text strings
--
CREATE OR REPLACE FUNCTION analyse_sentiment(data IN text[], model IN text DEFAULT NULL, label OUT text, score OUT float8)
    RETURNS SETOF record
    LANGUAGE sql
AS $BODY$
SELECT * FROM 
    json_to_recordset(
        marvin.run_pipeline(data, 'sentiment-analysis', model => model)
    ) AS x(label text, score float);
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.analyse_sentiment(text[], text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.analyse_sentiment(text[], text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.analyse_sentiment(text[], text) IS 'Analyse the sentiment of an array of text strings.';


--
-- Analyse the sentiment of a text string
--
CREATE OR REPLACE FUNCTION analyse_sentiment(data IN text, model IN text DEFAULT NULL, label OUT text, score OUT float8)
    RETURNS record
    LANGUAGE sql
AS $BODY$
SELECT * FROM 
    json_to_recordset(
        marvin.run_pipeline(ARRAY[data], 'sentiment-analysis', model => model)
    ) AS x(label text, score float) LIMIT 1;
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.analyse_sentiment(text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.analyse_sentiment(text, text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.analyse_sentiment(text, text) IS 'Analyse the sentiment of a text string.';
