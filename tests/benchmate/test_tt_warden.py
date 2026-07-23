from unittest.mock import patch

from benchmate.warden import TenstorrentWarden, gpu_warden, tt_warden


@patch("benchmate.warden._tt_smi_path", return_value="/usr/bin/tt-smi")
@patch("benchmate.warden.subprocess.run")
def test_tt_warden_resets_on_enter_and_exit(mock_run, _mock_path):
    with tt_warden():
        pass

    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/tt-smi", "-r"]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/tt-smi", "-r"]


@patch("benchmate.warden._tt_smi_path", return_value="/usr/bin/tt-smi")
@patch("benchmate.warden.subprocess.run")
def test_tt_warden_respects_reset_flags(mock_run, _mock_path):
    with TenstorrentWarden(reset_on_start=False, reset_on_end=True):
        pass

    assert mock_run.call_count == 1


@patch("benchmate.warden._tt_smi_path", return_value=None)
@patch("benchmate.warden.subprocess.run")
def test_tt_warden_skips_when_tt_smi_missing(mock_run, _mock_path):
    with tt_warden():
        pass

    mock_run.assert_not_called()


@patch("benchmate.warden.safe_get_gpu_info", return_value={"arch": "tt"})
@patch("benchmate.warden._tt_smi_path", return_value="/usr/bin/tt-smi")
@patch("benchmate.warden.subprocess.run")
def test_gpu_warden_uses_tt_warden_for_tt_arch(mock_run, _mock_path, _mock_gpu):
    with gpu_warden():
        pass

    assert mock_run.call_count == 2
