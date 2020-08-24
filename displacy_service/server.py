#!/usr/bin/env python
import falcon
import spacy
import json
import os
import logging
from logstash_formatter import LogstashFormatter
from prometheus_client import REGISTRY, generate_latest, Summary

from spacy.symbols import ENT_TYPE, TAG, DEP
import spacy.about
import spacy.util

from .parse import Parse, Entities, Sentences, SentencesDependencies

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s")
log = logging.getLogger()
if os.getenv('REWE_BDP_STAGE') is not None:
    loghandler = logging.StreamHandler()
    loghandler.setFormatter(LogstashFormatter())
    log.handlers = []
    log.addHandler(loghandler)

MODELS = os.getenv("languages", "").split()

# Create a metric to track time spent and requests made.
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request', labelnames=['method', 'endpoint'])

_models = {}


def get_model(model_name):
    if model_name not in _models:
        _models[model_name] = spacy.load(model_name)
    return _models[model_name]


def get_dep_types(model):
    '''List the available dep labels in the model.'''
    labels = []
    for label_id in model.parser.moves.freqs[DEP]:
        labels.append(model.vocab.strings[label_id])
    return labels


def get_ent_types(model):
    '''List the available entity types in the model.'''
    labels = []
    for label_id in model.entity.moves.freqs[ENT_TYPE]:
        labels.append(model.vocab.strings[label_id])
    return labels


def get_pos_types(model):
    '''List the available part-of-speech tags in the model.'''
    labels = []
    for label_id in model.tagger.moves.freqs[TAG]:
        labels.append(model.vocab.strings[label_id])
    return labels


def decode_request_body(req):
    req_body = req.bounded_stream.read()
    try:
        return json.loads(req_body.decode('utf8'))
    except Exception as e:
        log.error(f"Error decoding json {req_body}", exc_info=True)
        raise falcon.HTTPBadRequest(f'Decoding json failed: {req_body}')


class ModelsResource(object):
    """List the available models.

    test with: curl -s localhost:8000/models
    """

    def on_get(self, req, resp):
        try:
            output = list(MODELS)
            resp.body = json.dumps(output, sort_keys=True, indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Models retrieval failed!', exc_info=True)
            raise falcon.HTTPInternalServerError(
                'Models retrieval failed',
                '{}'.format(e))


class VersionResource(object):
    """Return the used spacy / api version

    test with: curl -s localhost:8000/version
    """

    def on_get(self, req, resp):
        try:
            resp.body = json.dumps({
                "spacy": spacy.about.__version__
            }, sort_keys=True, indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Version retrieval failed!', exc_info=True)
            raise falcon.HTTPInternalServerError(
                'Version retrieval failed',
                '{}'.format(e))


class SchemaResource(object):
    """Describe the annotation scheme of a model.

    This does not appear to work with later spacy
    versions.
    """

    def on_get(self, req, resp, model_name):
        try:
            model = get_model(model_name)
            output = {
                'dep_types': get_dep_types(model),
                'ent_types': get_ent_types(model),
                'pos_types': get_pos_types(model)
            }

            resp.body = json.dumps(output, sort_keys=True, indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Schema construction failed!', exc_info=True)
            raise falcon.HTTPBadRequest(
                'Schema construction failed',
                '{}'.format(e))


class DepResource(object):
    """Parse text and return displacy's expected JSON output.

    test with: curl -s localhost:8000/dep -d '{"text":"Pastafarians are smarter than people with Coca Cola bottles."}'
    """

    ENDPOINT = '/dep'
    R_TIME = REQUEST_TIME.labels(method='POST', endpoint=ENDPOINT)

    @R_TIME.time()
    def on_post(self, req, resp):
        json_data = decode_request_body(req)
        text = json_data.get('text')
        model_name = json_data.get('model', 'en')
        collapse_punctuation = json_data.get('collapse_punctuation', True)
        collapse_phrases = json_data.get('collapse_phrases', True)

        try:
            model = get_model(model_name)
            parse = Parse(model, text, collapse_punctuation, collapse_phrases)
            resp.body = json.dumps(parse.to_json(), sort_keys=True, indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Dependency parsing failed!', exc_info=True)
            raise falcon.HTTPBadRequest(
                'Dependency parsing failed',
                '{}'.format(e))


class EntResource(object):
    """Parse text and return displaCy ent's expected output."""
    ENDPOINT = '/ent'
    R_TIME = REQUEST_TIME.labels(method='POST', endpoint=ENDPOINT)

    @R_TIME.time()
    def on_post(self, req, resp):
        json_data = decode_request_body(req)
        text = json_data.get('text')
        model_name = json_data.get('model', 'en')
        log.debug(f"Requesting ENT for '{text}' with model {model_name}")
        try:
            model = get_model(model_name)
            entities = Entities(model, text)
            resp.body = json.dumps(entities.to_json(), sort_keys=True,
                                   indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Text parsing failed!', exc_info=True)
            raise falcon.HTTPBadRequest(
                'Text parsing failed',
                '{}'.format(e))


class SentsResources(object):
    """Returns sentences"""

    ENDPOINT = '/sents'
    R_TIME = REQUEST_TIME.labels(method='POST', endpoint=ENDPOINT)

    @R_TIME.time()
    def on_post(self, req, resp):
        json_data = decode_request_body(req)
        text = json_data.get('text')
        model_name = json_data.get('model', 'en')

        try:
            model = get_model(model_name)
            sentences = Sentences(model, text)
            resp.body = json.dumps(sentences.to_json(), sort_keys=True,
                                   indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Sentence tokenization failed!', exc_info=True)
            raise falcon.HTTPBadRequest(
                'Sentence tokenization failed',
                '{}'.format(e))


class SentsDepResources(object):
    """Returns sentences and dependency parses"""
    ENDPOINT = '/sents_dep'
    R_TIME = REQUEST_TIME.labels(method='POST', endpoint=ENDPOINT)

    @R_TIME.time()
    def on_post(self, req, resp):
        json_data = decode_request_body(req)
        text = json_data.get('text')
        model_name = json_data.get('model', 'en')
        collapse_punctuation = json_data.get('collapse_punctuation', False)
        collapse_phrases = json_data.get('collapse_phrases', False)

        try:
            model = get_model(model_name)
            sentences = SentencesDependencies(model,
                                              text,
                                              collapse_punctuation=collapse_punctuation,
                                              collapse_phrases=collapse_phrases)

            resp.body = json.dumps(sentences.to_json(),
                                   sort_keys=True,
                                   indent=2)
            resp.content_type = falcon.MEDIA_JSON
            resp.append_header('Access-Control-Allow-Origin', "*")
            resp.status = falcon.HTTP_200
        except Exception as e:
            log.error('Sentence tokenization and Dependency parsing failed!', exc_info=True)
            raise falcon.HTTPBadRequest(
                'Sentence tokenization and Dependency parsing failed',
                '{}'.format(e))


class MetricsResources(object):
    registry = REGISTRY

    def on_get(self, req, resp):
        registry = self.registry
        # Bake output
        output = generate_latest(registry)
        # Return output
        resp.body = output
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_TEXT


APP = falcon.API()
APP.add_route(DepResource.ENDPOINT, DepResource())
APP.add_route(EntResource.ENDPOINT, EntResource())
APP.add_route(SentsResources.ENDPOINT, SentsResources())
APP.add_route(SentsDepResources.ENDPOINT, SentsDepResources())
APP.add_route('/{model_name}/schema', SchemaResource())
APP.add_route('/models', ModelsResource())
APP.add_route('/version', VersionResource())
APP.add_route('/admin/prometheus', MetricsResources())
