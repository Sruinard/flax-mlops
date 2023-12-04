"""
This file contains the input pipeline for the trainer.
We'll use this to load the data and prepare it for training.
Note that there are two components to this file:
1. pipeline preparation (i.e convert to TFRecords)
2. 
"""

from collections import defaultdict
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class Playlist:
    tracks: list[str]
    albums: list[str]
    artists: list[str]

    @classmethod
    def from_json(cls, json):
        return cls(
            tracks=[t["track_uri"] for t in json["tracks"]],
            albums=[t["album_uri"] for t in json["tracks"]],
            artists=[t["artist_uri"] for t in json["tracks"]],
        )


def playlist_tfrecord_from_directory(src, dst):
    pass


def build_vocabs_from_playlist(playlists: list[Playlist]):
    KEYS = ["tracks", "albums", "artists"]
    vocabs = defaultdict(set)
    # triple nested loop. ouch.
    for playlist in playlists:
        for key in KEYS:
            for item in getattr(playlist, key):
                vocabs[key].add(item)

    lookup_layers = {}
    for key in KEYS:
        lookup_layer = tf.keras.layers.StringLookup()
        lookup_layer.adapt(sorted(vocabs[key]))
        lookup_layers[key] = lookup_layer

    return lookup_layers


def example_from_playlist():
    pass
