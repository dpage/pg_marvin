--
-- Create the virtual environment
--
CREATE OR REPLACE FUNCTION @extschema@.create_venv()
    RETURNS boolean
    LANGUAGE sql
AS $BODY$
    SELECT plpy_venv.create_venv('marvin');
    SELECT plpy_venv.activate_venv('marvin');
    SELECT plpy_venv.pip_install('{transformers[torch], Pillow, sentencepiece}');
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.create_venv() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.create_venv() TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.create_venv() IS 'Create a virtual environment for running pg_marvin.';


--
-- Run a pipeline task
--
CREATE OR REPLACE FUNCTION @extschema@.run_pipeline(data IN text[], task IN text, model IN text DEFAULT NULL)
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
CREATE OR REPLACE FUNCTION @extschema@.analyse_sentiment(data IN text[], model IN text DEFAULT NULL, label OUT text, score OUT float8)
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
CREATE OR REPLACE FUNCTION @extschema@.analyse_sentiment(data IN text, model IN text DEFAULT NULL, label OUT text, score OUT float8)
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


--
-- Translate an array of text strings from one language to another
--
CREATE OR REPLACE FUNCTION @extschema@.translate_text(data IN text[], source_lang IN text, target_lang IN text, model IN text DEFAULT NULL, translation OUT text)
    RETURNS SETOF text
    LANGUAGE sql
AS $BODY$
SELECT * FROM 
    json_to_recordset(
        marvin.run_pipeline(data, 'translation_' || source_lang || '_to_' || target_lang, model => model)
    ) AS x(translation_text text);
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.translate_text(text[], text, text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.translate_text(text[], text, text, text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.translate_text(text[], text, text, text) IS 'Translate an array of text strings from one language to another.';


--
-- Translate a text string from one language to another
--
CREATE OR REPLACE FUNCTION @extschema@.translate_text(data IN text, source_lang IN text, target_lang IN text, model IN text DEFAULT NULL, translation OUT text)
    RETURNS text
    LANGUAGE sql
AS $BODY$
SELECT * FROM 
    json_to_recordset(
        marvin.run_pipeline(ARRAY[data], 'translation_' || source_lang || '_to_' || target_lang, model => model)
    ) AS x(translation_text text) LIMIT 1;
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.translate_text(text, text, text, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.translate_text(text, text, text, text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.translate_text(text, text, text, text) IS 'Translate a text string from one language to another.';

--
-- Classify an image from bytea data (internal)
--
CREATE OR REPLACE function @extschema@.classify_image(image IN bytea, model IN text, label OUT text, score OUT float)
    RETURNS record
    LANGUAGE plpython3u
AS $BODY$
from PIL import Image
from transformers import pipeline
import io
import json

img_data = Image.open(io.BytesIO(image))
vision_classifier = pipeline("image-classification", model=model, framework='pt')

preds = vision_classifier(images=img_data)

score = 0
final = {}

for pred in preds:
    if pred['score'] > score:
        score = pred['score']
        final = pred

return final
$BODY$;

REVOKE ALL PRIVILEGES ON FUNCTION @extschema@.classify_image(bytea, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION @extschema@.classify_image(bytea, text) TO CURRENT_USER;
COMMENT ON FUNCTION @extschema@.classify_image(bytea, text) IS 'Classify an image from bytea data.';
