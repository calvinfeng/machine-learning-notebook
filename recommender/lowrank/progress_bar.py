import sys
import math


class ProgressBar(object):
    def __init__(self, title, N):
        """
        :param string title
        :param int N: Total number of iterations
        """
        self.title = title
        self.num_iters = N
        self.interval = int(round(float(N) / 100))

    def describe(self, msg):
        print u'\u25cc' + msg

    def report(self, i, curr_cost):
        if i % self.interval == 0:
            percentage = round(100 * i / self.num_iters)
            progress = int(math.floor(percentage / 5)) * u'\u2588'
            check_box = u'\u2610'
            sys.stdout.write('\r%s %s: |%-20s| %d%% - current cost: %s' % (
                check_box, self.title, progress, percentage, curr_cost))
            sys.stdout.flush()

    def complete(self):
        percentage = 100
        progress = 20 * u'\u2588'
        check_mark = u'\u2611'
        sys.stdout.write('\r%s %s: |%-20s| %d%%\n\n' % (check_mark, self.title, progress, percentage))
        sys.stdout.flush()
