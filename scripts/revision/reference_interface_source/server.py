from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import argparse,json
from predictor import predict
ROOT=Path(__file__).resolve().parent
class Handler(BaseHTTPRequestHandler):
    def send(self,status,data,ctype):
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        routes={'/':('index.html','text/html; charset=utf-8'),'/history-template.csv':('examples/history_168h.csv','text/csv; charset=utf-8'),'/recorded-feed-example.csv':('examples/recorded_feed_oracle_24h.csv','text/csv; charset=utf-8')}
        if self.path=='/health':return self.send(200,b'{"ok":true,"app":"Muscatine biogas predictor"}','application/json')
        if self.path not in routes:return self.send(404,b'Not found','text/plain')
        file,ctype=routes[self.path];self.send(200,(ROOT/file).read_bytes(),ctype)
    def do_POST(self):
        if self.path!='/predict':return self.send(404,b'Not found','text/plain')
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<1 or n>5_000_000:raise ValueError('Upload limit: 5 MB.')
            body=json.loads(self.rfile.read(n)); result=predict(body['history'],body.get('plan'))
            self.send(200,json.dumps(result,allow_nan=False).encode(),'application/json; charset=utf-8')
        except (ValueError,KeyError,TypeError) as e:self.send(400,json.dumps({'error':str(e)}).encode(),'application/json; charset=utf-8')
    def log_message(self,*args):pass
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8765);a=ap.parse_args()
    print(f'GUI available at http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
