import base64
import binascii
import hashlib
import io
import json
import os
import urllib.request
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
QUESTION_TYPES = ("noul", "choice", "score")
MODEL_AND_STATE_REQUIRED = "model and state are required"
AT_LEAST_ONE_QUESTION = "at least one question is required"
REMOTE_TIMEOUT_S = 10.0
MAX_IMAGE_BYTES = 20 << 20
DATA_URL_PREFIX = "data:"


def question_type_message(question_id):
    return f"{question_id}: type must be noul, choice, or score"


def criteria_message(question_id):
    return f"{question_id}: criteria must not be empty"


class Noul(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] = Field(min_length=1)


class Score(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Union[Noul, Choice, Score]


def check_release_rules(body):
    if not isinstance(body, dict):
        return body
    questions = body.get("questions")
    if not isinstance(body.get("model"), str) or "state" not in body:
        raise ValueError(MODEL_AND_STATE_REQUIRED)
    if not isinstance(questions, dict) or not questions:
        raise ValueError(AT_LEAST_ONE_QUESTION)
    for question_id, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            raise ValueError(question_type_message(question_id))
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(criteria_message(question_id))
        if question["type"] == "choice" and not isinstance(question["criteria"], dict):
            raise ValueError(f"{question_id}: choice criteria must be a mapping of option id to description")
        if question["type"] == "score" and not isinstance(question["criteria"], list):
            raise ValueError(f"{question_id}: score criteria must be a list of ordered option descriptions")
        noul_criteria = question.get("criteria") if question["type"] == "noul" else None
        if noul_criteria is not None and not isinstance(noul_criteria, dict):
            raise ValueError(
                f"{question_id}: noul criteria must be a mapping with optional true and false descriptions"
            )
    return body


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)
    images: list[str] | None = None
    videos: list[list[str]] | None = None
    media_kwargs: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _release_rules(cls, body):
        return check_release_rules(body)

    def question_dicts(self):
        out = {}
        for question_id, question in self.questions.items():
            item = {"type": question.type}
            if question.instructions is not None:
                item["instructions"] = question.instructions
            if question.criteria is not None:
                item["criteria"] = question.criteria
            out[question_id] = item
        return out

    def has_media(self):
        return bool(self.images) or bool(self.videos)


def api_request(record):
    body = {key: record[key] for key in ("model", "state", "images", "videos", "media_kwargs") if key in record}
    body.setdefault("model", "clef")
    body["questions"] = {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
        for qid, q in record["questions"].items()
    }
    return body


def allow_remote_images():
    return os.environ.get("CLEF_ALLOW_REMOTE_IMAGES", "1") == "1"


def _fetch_remote(url, label):
    if not allow_remote_images():
        raise ValueError(f"{label}: remote image URLs are disabled on this server (CLEF_ALLOW_REMOTE_IMAGES=0)")
    request = urllib.request.Request(url, headers={"user-agent": "clef-tt-server"})
    try:
        with urllib.request.urlopen(request, timeout=REMOTE_TIMEOUT_S) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except Exception as error:
        raise ValueError(f"{label}: could not fetch {url}: {error}") from error
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label}: image over the {MAX_IMAGE_BYTES >> 20} MiB limit")
    return data


def image_bytes(source, label):
    if not isinstance(source, str) or not source:
        raise ValueError(f"{label}: expected a base64 string, a data: URL or an http(s) URL")
    if source.startswith(("http://", "https://")):
        return _fetch_remote(source, label)
    payload = source
    if source.startswith(DATA_URL_PREFIX):
        header, sep, payload = source.partition(",")
        if not sep or ";base64" not in header:
            raise ValueError(f"{label}: data: URL must be base64 encoded")
    if len(payload) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        raise ValueError(f"{label}: image over the {MAX_IMAGE_BYTES >> 20} MiB limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{label}: invalid base64 image data") from error
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label}: image over the {MAX_IMAGE_BYTES >> 20} MiB limit")
    return data


def decode_image(data, label):
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"{label}: not a decodable image ({error})") from error
    return image.convert("RGB")


def decode_media(req):
    images, videos = [], []
    digest = hashlib.sha1()
    for index, source in enumerate(req.images or []):
        data = image_bytes(source, f"images[{index}]")
        digest.update(b"i")
        digest.update(hashlib.sha1(data).digest())
        images.append(decode_image(data, f"images[{index}]"))
    for vindex, frames in enumerate(req.videos or []):
        if not frames:
            raise ValueError(f"videos[{vindex}]: a video needs at least one frame")
        decoded = []
        digest.update(b"v")
        for findex, source in enumerate(frames):
            data = image_bytes(source, f"videos[{vindex}][{findex}]")
            digest.update(hashlib.sha1(data).digest())
            decoded.append(decode_image(data, f"videos[{vindex}][{findex}]"))
        videos.append(decoded)
    if (images or videos) and req.media_kwargs:
        digest.update(b"k")
        digest.update(json.dumps(req.media_kwargs, sort_keys=True, separators=(",", ":"), default=str).encode())
    media_digest = digest.hexdigest() if images or videos else None
    return images, videos, media_digest


def to_record(req, images=None, videos=None):
    record = {"model": req.model, "state": req.state, "questions": req.question_dicts()}
    if images:
        record["images"] = images
    if videos:
        record["videos"] = videos
    if req.media_kwargs:
        record["media_kwargs"] = req.media_kwargs
    return record
