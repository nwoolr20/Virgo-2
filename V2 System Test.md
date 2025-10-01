# V2 System Test

**Test Date:** 2025-10-01 01:49:05
**Test Duration:** 63.66 seconds

This document contains the complete results of the Virgo-2 system test,
following the steps outlined in the QUICKSTART guide.

---


## Step 1: Install Package

Upgrading pip, setuptools, and wheel...
Running: Upgrade pip, setuptools, and wheel
**Command:** `pip install --upgrade pip setuptools wheel`

**Output:**
```
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /home/runner/.local/lib/python3.12/site-packages (25.2)
Requirement already satisfied: setuptools in /home/runner/.local/lib/python3.12/site-packages (80.9.0)
Requirement already satisfied: wheel in /home/runner/.local/lib/python3.12/site-packages (0.45.1)

```

**Elapsed Time:** 0.78 seconds
**Exit Code:** 0
✓ Upgrade pip, setuptools, and wheel - SUCCESS
**Status:** ✓ SUCCESS

Installing dependencies...
Running: Install Python dependencies
**Command:** `pip install torch sentence-transformers faiss-cpu scikit-learn textblob pytest nltk`

**Output:**
```
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: torch in /home/runner/.local/lib/python3.12/site-packages (2.8.0)
Requirement already satisfied: sentence-transformers in /home/runner/.local/lib/python3.12/site-packages (5.1.1)
Requirement already satisfied: faiss-cpu in /home/runner/.local/lib/python3.12/site-packages (1.12.0)
Requirement already satisfied: scikit-learn in /home/runner/.local/lib/python3.12/site-packages (1.7.2)
Requirement already satisfied: textblob in /home/runner/.local/lib/python3.12/site-packages (0.19.0)
Requirement already satisfied: pytest in /home/runner/.local/lib/python3.12/site-packages (8.4.2)
Requirement already satisfied: nltk in /home/runner/.local/lib/python3.12/site-packages (3.9.1)
Requirement already satisfied: filelock in /home/runner/.local/lib/python3.12/site-packages (from torch) (3.19.1)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/lib/python3/dist-packages (from torch) (4.10.0)
Requirement already satisfied: setuptools in /home/runner/.local/lib/python3.12/site-packages (from torch) (80.9.0)
Requirement already satisfied: sympy>=1.13.3 in /home/runner/.local/lib/python3.12/site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx in /home/runner/.local/lib/python3.12/site-packages (from torch) (3.5)
Requirement already satisfied: jinja2 in /usr/lib/python3/dist-packages (from torch) (3.1.2)
Requirement already satisfied: fsspec in /home/runner/.local/lib/python3.12/site-packages (from torch) (2025.9.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /home/runner/.local/lib/python3.12/site-packages (from torch) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /home/runner/.local/lib/python3.12/site-packages (from torch) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /home/runner/.local/lib/python3.12/site-packages (from torch) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /home/runner/.local/lib/python3.12/site-packages (from torch) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /home/runner/.local/lib/python3.12/site-packages (from torch) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /home/runner/.local/lib/python3.12/site-packages (from torch) (2.27.3)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /home/runner/.local/lib/python3.12/site-packages (from torch) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /home/runner/.local/lib/python3.12/site-packages (from torch) (1.13.1.3)
Requirement already satisfied: triton==3.4.0 in /home/runner/.local/lib/python3.12/site-packages (from torch) (3.4.0)
Requirement already satisfied: transformers<5.0.0,>=4.41.0 in /home/runner/.local/lib/python3.12/site-packages (from sentence-transformers) (4.56.2)
Requirement already satisfied: tqdm in /home/runner/.local/lib/python3.12/site-packages (from sentence-transformers) (4.67.1)
Requirement already satisfied: scipy in /home/runner/.local/lib/python3.12/site-packages (from sentence-transformers) (1.16.2)
Requirement already satisfied: huggingface-hub>=0.20.0 in /home/runner/.local/lib/python3.12/site-packages (from sentence-transformers) (0.35.3)
Requirement already satisfied: Pillow in /home/runner/.local/lib/python3.12/site-packages (from sentence-transformers) (11.3.0)
Requirement already satisfied: numpy>=1.17 in /home/runner/.local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2.3.3)
Requirement already satisfied: packaging>=20.0 in /usr/lib/python3/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (24.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/lib/python3/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (6.0.1)
Requirement already satisfied: regex!=2019.12.17 in /home/runner/.local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2025.9.18)
Requirement already satisfied: requests in /usr/lib/python3/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2.31.0)
Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /home/runner/.local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.22.1)
Requirement already satisfied: safetensors>=0.4.3 in /home/runner/.local/lib/python3.12/site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.6.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /home/runner/.local/lib/python3.12/site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.1.10)
Requirement already satisfied: joblib>=1.2.0 in /home/runner/.local/lib/python3.12/site-packages (from scikit-learn) (1.5.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/runner/.local/lib/python3.12/site-packages (from scikit-learn) (3.6.0)
Requirement already satisfied: iniconfig>=1 in /home/runner/.local/lib/python3.12/site-packages (from pytest) (2.1.0)
Requirement already satisfied: pluggy<2,>=1.5 in /home/runner/.local/lib/python3.12/site-packages (from pytest) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /usr/lib/python3/dist-packages (from pytest) (2.17.2)
Requirement already satisfied: click in /usr/lib/python3/dist-packages (from nltk) (8.1.6)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/runner/.local/lib/python3.12/site-packages (from sympy>=1.13.3->torch) (1.3.0)

```

**Elapsed Time:** 0.68 seconds
**Exit Code:** 0
✓ Install Python dependencies - SUCCESS
**Status:** ✓ SUCCESS

Running: Install package in editable mode
**Command:** `pip install -e .`

**Output:**
```
Defaulting to user installation because normal site-packages is not writeable
Obtaining file:///home/runner/work/Virgo-2/Virgo-2
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'

```

**Errors/Warnings:**
```
ERROR: Exception:
Traceback (most recent call last):
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/cli/base_command.py", line 107, in _run_wrapper
    status = _inner_run()
             ^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/cli/base_command.py", line 98, in _inner_run
    return self.run(options, args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/cli/req_command.py", line 71, in wrapper
    return func(self, options, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/commands/install.py", line 393, in run
    requirement_set = resolver.resolve(
                      ^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 79, in resolve
    collected = self.factory.collect_root_requirements(root_reqs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 538, in collect_root_requirements
    reqs = list(
           ^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 494, in _make_requirements_from_install_req
    cand = self._make_base_candidate_from_link(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 205, in _make_base_candidate_from_link
    self._editable_candidate_cache[link] = EditableCandidate(
                                           ^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 334, in __init__
    super().__init__(
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 162, in __init__
    self.dist = self._prepare()
                ^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 239, in _prepare
    dist = self._prepare_distribution()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 344, in _prepare_distribution
    return self._factory.preparer.prepare_editable_requirement(self._ireq)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 708, in prepare_editable_requirement
    dist = _get_prepared_distribution(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 77, in _get_prepared_distribution
    abstract_dist.prepare_distribution_metadata(
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 57, in prepare_distribution_metadata
    self.req.isolated_editable_sanity_check()
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/req/req_install.py", line 544, in isolated_editable_sanity_check
    and not self.supports_pyproject_editable
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/functools.py", line 995, in __get__
    val = self.func(instance)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_internal/req/req_install.py", line 261, in supports_pyproject_editable
    return "build_editable" in self.pep517_backend._supported_features()
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 180, in _supported_features
    return self._call_hook("_supported_features", {})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/.local/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 402, in _call_hook
    raise BackendUnavailable(
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_backend'

```

**Elapsed Time:** 1.48 seconds
**Exit Code:** 2
✗ Install package in editable mode - FAILED (exit code 2)
**Status:** ✗ FAILED

Installation failed, continuing anyway...

## Step 2: Download Required Data

Running: Download NLTK data
**Command:** `python -c "import nltk; nltk.download('brown', quiet=True); nltk.download('punkt', quiet=True); print('✓ NLTK data downloaded')"`

**Output:**
```
✓ NLTK data downloaded

```

**Elapsed Time:** 0.94 seconds
**Exit Code:** 0
✓ Download NLTK data - SUCCESS
**Status:** ✓ SUCCESS

Running: Download sentence-transformers model
**Command:** `python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('✓ Sentence-transformers model loaded')"`

**Output:**
```
✓ Sentence-transformers model loaded

```

**Elapsed Time:** 4.71 seconds
**Exit Code:** 0
✓ Download sentence-transformers model - SUCCESS
**Status:** ✓ SUCCESS


## Step 3: Verify Installation with Test Suite

Running: Run test suite
**Command:** `pytest tests/ -v`

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/runner/work/Virgo-2/Virgo-2
configfile: pyproject.toml
collecting ... collected 12 items

tests/test_coordinates.py::test_coordinate_dimensions PASSED             [  8%]
tests/test_coordinates.py::test_importance_scoring PASSED                [ 16%]
tests/test_coordinates.py::test_semantic_consistency PASSED              [ 25%]
tests/test_field.py::test_field_forward PASSED                           [ 33%]
tests/test_field.py::test_field_overfit FAILED                           [ 41%]
tests/test_field.py::test_field_batch PASSED                             [ 50%]
tests/test_integration.py::test_end_to_end PASSED                        [ 58%]
tests/test_integration.py::test_persistence_end_to_end PASSED            [ 66%]
tests/test_memory.py::test_store_retrieve PASSED                         [ 75%]
tests/test_memory.py::test_field_training PASSED                         [ 83%]
tests/test_memory.py::test_persistence PASSED                            [ 91%]
tests/test_memory.py::test_stats PASSED                                  [100%]

=================================== FAILURES ===================================
______________________________ test_field_overfit ______________________________

    def test_field_overfit():
        """Field should memorize single mapping."""
        field = ConversationField(hidden_dim=128)
    
        coord = torch.rand(1, 6)
        target = torch.rand(1, 384)
    
        metrics = field.fit_memory(coord, target, num_steps=1000, lr=1e-3, verbose=False)
    
>       assert metrics["losses"][-1] < 0.01
E       assert 0.036005377769470215 < 0.01

tests/test_field.py:26: AssertionError
=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_field.py::test_field_overfit - assert 0.036005377769470215 < 0.01
================== 1 failed, 11 passed, 3 warnings in 14.39s ===================

```

**Elapsed Time:** 15.85 seconds
**Exit Code:** 1
✗ Run test suite - FAILED (exit code 1)
**Status:** ✗ FAILED


## Step 4: Run Interactive Demo

Running: Run demo script
**Command:** `python scripts/demo.py`

**Output:**
```
============================================================
Neural Field Memory System - Demo
============================================================

Adding initial memories...
  Stored: My name is Alice
  Stored: I work as a software engineer at Google
  Stored: I have two cats named Whiskers and Mittens
  Stored: My favorite programming language is Python
  Stored: I live in San Francisco

Training neural field...
✓ Field trained

============================================================
Testing Retrieval
============================================================

Query: What is my name?
Top results:
  1. [dist=0.64] My name is Alice
  2. [dist=1.54] I have two cats named Whiskers and Mittens

Query: Where do I work?
Top results:
  1. [dist=1.19] I work as a software engineer at Google
  2. [dist=1.40] I live in San Francisco

Query: Tell me about my pets
Top results:
  1. [dist=1.04] I have two cats named Whiskers and Mittens
  2. [dist=1.52] My name is Alice

Query: What programming language do I like?
Top results:
  1. [dist=0.38] My favorite programming language is Python
  2. [dist=1.48] I work as a software engineer at Google

Query: Where do I live?
Top results:
  1. [dist=1.00] I live in San Francisco
  2. [dist=1.63] I work as a software engineer at Google

============================================================
System Statistics
============================================================
  total_memories: 5
  user_turns: 5
  assistant_turns: 0
  is_fitted: True
  has_pca: True

============================================================
Demo Complete!
============================================================


```

**Elapsed Time:** 6.34 seconds
**Exit Code:** 0
✓ Run demo script - SUCCESS
**Status:** ✓ SUCCESS


## Step 5: Run Comprehensive Evaluation

Running: Run evaluation script
**Command:** `python scripts/evaluate.py`

**Output:**
```

============================================================
NEURAL FIELD SYSTEM EVALUATION
============================================================

============================================================
COMPRESSION EVALUATION
============================================================
Training field on 300 memories...
Step 0/3000, Loss: 0.004888
Step 500/3000, Loss: 0.000001
Step 1000/3000, Loss: 0.000000
Step 1500/3000, Loss: 0.000000
Step 2000/3000, Loss: 0.000000
Step 2500/3000, Loss: 0.000000
Final loss: 0.000000

=== Storage Comparison ===
Memories stored: 300
Raw JSON: 16,300 bytes
Gzipped JSON: 275 bytes
Neural Field: 3,992,251 bytes

Compression ratio (vs JSON): 0.00x
Compression ratio (vs gzip): 0.00x

=== Retrieval Tests ===
✓ Query: 'What is my name?' - Expected: 'Alice'
✗ Query: 'What are my cats called?' - Expected: 'Whiskers'
✗ Query: 'What is my job?' - Expected: 'engineer'

Retrieval accuracy: 33.3%

============================================================
PERSISTENCE EVALUATION
============================================================

Phase 1: Creating and saving system...
✓ System saved

Phase 2: Loading in new instance...
✓ System loaded
✓ Verified 3 memories
✓ Field fitted: True

Testing retrieval after load...
✓ Query: 'What color do I like?' - Expected: 'blue'
✓ Query: 'Where do I live?' - Expected: 'Seattle'

✓ Persistence test PASSED

============================================================
EVALUATION SUMMARY
============================================================

Compression Results:
  vs JSON:  0.00x
  vs Gzip:  0.00x
  Retrieval: 33.3%

Persistence:
  ✓ PASSED

============================================================
SUCCESS CRITERIA
============================================================
  Compression vs Gzip > 1.5x: ✗ FAIL
  Retrieval Accuracy > 75%:   ✗ FAIL
  Persistence:                ✓ PASS

============================================================
✗ OVERALL: SYSTEM NEEDS IMPROVEMENT
============================================================


```

**Elapsed Time:** 26.03 seconds
**Exit Code:** 0
✓ Run evaluation script - SUCCESS
**Status:** ✓ SUCCESS


## Step 6: Try Chat Interface (Automated Test)

Running: Automated chat interface test
**Command:** `python /tmp/test_chat_interface.py`

**Output:**
```
Using temporary memory store: /tmp/tmp1_jayb61

=== Initializing Memory System ===

=== Adding Test Memories ===
  1. Stored: My name is Alice
  2. Stored: I love programming in Python
  3. Stored: I work as a software engineer
  4. Stored: My favorite color is blue
  5. Stored: I have a cat named Whiskers

=== Training Neural Field ===
✓ Field trained successfully

=== Testing Retrieval ===

Query: What is my name?
Results:
  1. [distance=0.64] My name is Alice
  2. [distance=1.44] I have a cat named Whiskers

Query: What programming language do I like?
Results:
  1. [distance=0.80] I love programming in Python
  2. [distance=1.36] I work as a software engineer

Query: Tell me about my pet
Results:
  1. [distance=1.07] I have a cat named Whiskers
  2. [distance=1.55] My name is Alice

=== System Statistics ===
  total_memories: 5
  user_turns: 5
  assistant_turns: 0
  is_fitted: True
  has_pca: True

=== Saving System ===
✓ System saved to /tmp/tmp1_jayb61

=== Loading System ===
✓ System loaded successfully

=== Verifying Loaded System ===
Loaded memories: 5
Field fitted: True

✓ Cleaned up temporary directory

=== Chat Interface Test Complete ===

```

**Elapsed Time:** 6.85 seconds
**Exit Code:** 0
✓ Automated chat interface test - SUCCESS
**Status:** ✓ SUCCESS


## Test Summary


**Total Steps:** 6
**Passed:** 4 ✓
**Failed:** 2 ✗
**Success Rate:** 66.7%

### Detailed Results:

- Step 1: Install Package: ✗ FAIL
- Step 2: Download Required Data: ✓ PASS
- Step 3: Verify Installation: ✗ FAIL
- Step 4: Run Demo: ✓ PASS
- Step 5: Run Evaluation: ✓ PASS
- Step 6: Chat Interface Test: ✓ PASS

### Overall Status

**✗ SOME TESTS FAILED (2/6)**

✗ SOME TESTS FAILED (2/6)
**Total Test Duration:** 63.66 seconds


## Saving Results

