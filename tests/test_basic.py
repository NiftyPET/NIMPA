from os import path

from niftypet import nimpa


def test_dev_info(capsys):
    devs = nimpa.dev_info()
    out, err = capsys.readouterr()
    assert not any((out, err))
    nimpa.dev_info(showprop=True)
    out, err = capsys.readouterr()
    assert not err
    assert not devs or out


def test_resources():
    assert path.exists(nimpa.path_resources)
    assert nimpa.resources.DIRTOOLS
