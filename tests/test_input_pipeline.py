# this test checks if we can transform a list of playlists into a list of examples
import pytest
import tensorflow as tf
from trainer import input_pipeline as ip


# - convert json to playlist
# - create mapping for lookup uris from playlists
# - create tfrecord from playlist


@pytest.fixture
def playlist_file_content():
    return {
        "playlists": [
            {
                "tracks": [
                    {
                        "track_uri": "spotify:track:a",
                        "album_uri": "spotify:album:a",
                        "artist_uri": "spotify:artist:a",
                    },
                    {
                        "track_uri": "spotify:track:b",
                        "album_uri": "spotify:album:b",
                        "artist_uri": "spotify:artist:b",
                    },
                ]
            }
        ]
    }


def test_create_playlist_from_json(playlist_file_content):
    playlists = [
        ip.Playlist.from_json(playlist)
        for playlist in playlist_file_content["playlists"]
    ]
    playlist = playlists[0]
    assert len(playlist.tracks) == 2
    assert len(playlist.albums) == 2
    assert len(playlist.artists) == 2


def test_build_table_lookups(playlist_file_content):
    playlists = [
        ip.Playlist.from_json(playlist)
        for playlist in playlist_file_content["playlists"]
    ]

    # build the lookup tables and select the track vocab for testing
    track_vocab = ip.build_vocabs_from_playlist(playlists)["tracks"]

    # forward pass (uri -> index)
    uri_index = track_vocab(tf.constant(["spotify:track:a"]))
    missing_uri = track_vocab(tf.constant(["some:missing:uri"]))
    # uri_index non-zero means we have a hit
    assert uri_index != 0
    # uri_index zero means we have an unknown uri
    assert missing_uri == 0
