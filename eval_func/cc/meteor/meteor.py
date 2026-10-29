#!/usr/bin/env python

# Python wrapper for the METEOR java implementation.

import os
import shutil
import subprocess
import tempfile
import threading

# meteor-1.5.jar is expected next to this file.
METEOR_JAR = 'meteor-1.5.jar'
_JAR_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_dir():
    """Return an ASCII-only directory holding the jar, since METEOR 1.5 mishandles non-ASCII paths."""
    if _JAR_DIR.isascii():
        return _JAR_DIR

    mirror = os.path.join(tempfile.gettempdir(), 'rsicc_meteor')
    os.makedirs(mirror, exist_ok=True)

    jar_dst = os.path.join(mirror, METEOR_JAR)
    if not os.path.exists(jar_dst):
        shutil.copyfile(os.path.join(_JAR_DIR, METEOR_JAR), jar_dst)

    data_dst = os.path.join(mirror, 'data')
    data_src = os.path.join(_JAR_DIR, 'data')
    # replace a stale symlink pointing at a moved directory
    if os.path.lexists(data_dst) and os.path.realpath(data_dst) != os.path.realpath(data_src):
        os.unlink(data_dst)
    if not os.path.lexists(data_dst):
        os.symlink(data_src, data_dst)

    return mirror


class Meteor:

    def __init__(self):
        self.env = os.environ
        self.env['LC_ALL'] = 'en_US.UTF_8'
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                                         cwd=_run_dir(), \
                                         stdin=subprocess.PIPE, \
                                         stdout=subprocess.PIPE, \
                                         stderr=subprocess.PIPE,
                                         env=self.env, universal_newlines=True, bufsize=1)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):

        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in range(len(res)):
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        # Send to METEOR
        self.meteor_p.stdin.write(eval_line + '\n')

        # Collect segment scores
        for i in range(len(res)):
            score = float(self.meteor_p.stdout.readline().strip())
            scores.append(score)

        # Final score
        final_score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return final_score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(score_line + '\n')
        return self.meteor_p.stdout.readline().strip()

    def __del__(self):
        if not hasattr(self, 'lock'):
            return
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()