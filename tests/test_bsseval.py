import numpy as np
import pytest


# def __unit_test_empty_input(metric):
#     if (metric == mir_eval.separation.bss_eval_sources or
#             metric == mir_eval.separation.bss_eval_images):
#         args = [np.array([]), np.array([])]
#     elif (metric == mir_eval.separation.bss_eval_sources_framewise or
#             metric == mir_eval.separation.bss_eval_images_framewise):
#         args = [np.array([]), np.array([]), 40, 20]
#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter('always')
#         # First, test for a warning on empty audio data
#         metric(*args)
#         assert len(w) == 2
#         assert issubclass(w[-1].category, UserWarning)
#         # And that the metric returns empty arrays
#         assert np.allclose(metric(*args), np.array([]))
#
#
#
# def test_separation_functions():
#     assert len(ref_files) == len(est_files) == len(sco_files) > 0
#
#     # Unit tests
#
#     for metric in [mir_eval.separation.bss_eval_sources,
#                    mir_eval.separation.bss_eval_sources_framewise,
#                    mir_eval.separation.bss_eval_images,
#                    mir_eval.separation.bss_eval_images_framewise]:
#         yield (__unit_test_empty_input, metric)
#         yield (__unit_test_silent_input, metric)
#         yield (__unit_test_incompatible_shapes, metric)
#         yield (__unit_test_too_many_sources, metric)
#         yield (__unit_test_too_many_dimensions, metric)
#     for metric in [mir_eval.separation.bss_eval_sources,
#                    mir_eval.separation.bss_eval_images]:
#         yield (__unit_test_default_permutation, metric)
#     for metric in [mir_eval.separation.bss_eval_sources_framewise,
#                    mir_eval.separation.bss_eval_images_framewise]:
#         yield (__unit_test_framewise_small_window, metric)
#         yield (__unit_test_partial_silence, metric)
