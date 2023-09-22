import logging
from logging import handlers

level_relations = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}


class Logger(logging.Logger):

    def __init__(
            self,
            log_path,
            name='root',
            level='info',
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        logging.Logger.__init__(self, name)

        self.setLevel(level_relations.get(level))

        # custom format
        format_str = logging.Formatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        self.addHandler(sh)

        filename = "%s/log.txt" % log_path
        th = handlers.RotatingFileHandler(filename=filename, encoding='utf-8')
        th.setFormatter(format_str)
        self.addHandler(th)

