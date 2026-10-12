"""Flask app: the page and its JSON API.

  GET  /                                  the page (static/index.html; refreshes every ~15 s)
  GET  /api/state                         everything: projects, runs, machine strip, pause-all
  POST /api/run/<project>/<run>/pause     create <run>/PAUSE ("by": "zerodash <client>")
  POST /api/run/<project>/<run>/resume    remove <run>/PAUSE
  POST /api/pause-all                     PAUSE every active run that honours it
  POST /api/resume-all                    remove only the PAUSE files a pause-all wrote

Every POST needs a JSON object body with "confirm": true (the page's confirm step; a plain
cross-site form cannot send JSON). Runs are looked up among the discovered runs only.
"""
import os

from flask import Flask, jsonify, request, send_from_directory

from . import PACKAGE_DIR
from .control import ControlError
from .model import NotFound
from .remote import SshError

STATIC = os.path.join(PACKAGE_DIR, 'static')


def create_app(dash):
    app = Flask(__name__, static_folder=STATIC, static_url_path='/static')
    app.json.sort_keys = False
    app.config['DASH'] = dash

    def confirmed():
        body = request.get_json(silent=True)
        return isinstance(body, dict) and body.get('confirm') is True

    def need_confirm():
        return jsonify({'ok': False, 'error': 'send a JSON body with "confirm": true'}), 400

    @app.after_request
    def no_cache(resp):
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    @app.route('/')
    def index():
        return send_from_directory(STATIC, 'index.html')

    @app.route('/api/state')
    def api_state():
        return jsonify(dash.snapshot())

    @app.route('/api/run/<pid>/<rid>/<action>', methods=['POST'])
    def api_run_action(pid, rid, action):
        if action not in ('pause', 'resume'):
            return jsonify({'ok': False, 'error': f'no action {action!r}'}), 404
        if not confirmed():
            return need_confirm()
        try:
            if action == 'pause':
                res = dash.pause(pid, rid, by=f'zerodash {request.remote_addr}')
            else:
                res = dash.resume(pid, rid)
        except NotFound:
            return jsonify({'ok': False, 'error': f'no run {pid}/{rid}'}), 404
        except ControlError as e:
            return jsonify({'ok': False, 'error': str(e)}), 409
        except SshError as e:
            return jsonify({'ok': False, 'error': str(e)}), 502
        return jsonify(res)

    @app.route('/api/pause-all', methods=['POST'])
    def api_pause_all():
        if not confirmed():
            return need_confirm()
        return jsonify(dash.pause_all())

    @app.route('/api/resume-all', methods=['POST'])
    def api_resume_all():
        if not confirmed():
            return need_confirm()
        return jsonify(dash.resume_all())

    @app.route('/healthz')
    def healthz():
        return jsonify({'ok': True})

    return app
