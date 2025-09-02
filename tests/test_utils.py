from osos.config_hashing import LoggingLevel, hash_config_and_data


def test_hash_config():
    from json import dumps
    from hashlib import sha256

    config = {"foo": "bar", "model": "baz"}

    hashed_result = hash_config_and_data(
        config, hash_git=False, logging_level=LoggingLevel.INFO
    )

    dumped_config = dumps(config, sort_keys=True).encode()
    expect = sha256(dumped_config).hexdigest()

    assert hashed_result.get("config") == expect
    assert hashed_result.get("hashed_files", None) is None
    assert hashed_result.get("git_info", None) is None

    # NOTE: This value was hard-coded by running the test failing once and copying the
    # result. The digest is sequentially constructed based on what we hash. Ways this
    # value could change would be
    # 1. If the implementation changes
    # 2. If a different hash function from sha256 __with `usedforsecurity=False`__ were
    #    modified in any way.
    # 3. If the arguments concerning what to hash changed.
    digest = hashed_result.get("digest", "")
    assert digest == "69f1c88ea689791e6d192ab19146433540abc3973a6981a6318771e0f956d34d"

    # This should always be stable. The sha256 algorithm encodes 256 bits
    # (64 chars x 4 bytes/char) of data in the result. As before, this part is basically
    # an invariant to how much of the digest is produced for the user.
    assert len(digest) == 64
