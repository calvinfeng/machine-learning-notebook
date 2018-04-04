from sys import stdout
from math import floor

class Progress:
    def __init__(self, title, total_iteration):
        self.title = title
        self.total_iteration = total_iteration
        self.interval = round(total_iteration / 100)

    def describe(self, text):
        print u'\u25cc ' + text

    def report(self, iteration, cost):
        if iteration % self.interval == 0:
            percentage = round(100 * iteration / self.total_iteration)
            progress_bar = int(floor(percentage / 5)) * u'\u2588'
            check_box = u'\u2610'
            stdout.write('\r%s %s: |%-20s| %d%% - current cost: %s' % (check_box, self.title, progress_bar, percentage, cost))
            stdout.flush()

    def complete(self):
        percentage = 100
        progress_bar = 20 * u'\u2588'
        check_mark = u'\u2611'
        stdout.write('\r%s %s: |%-20s| %d%%\n\n' % (check_mark, self.title, progress_bar, percentage))
        stdout.flush()
