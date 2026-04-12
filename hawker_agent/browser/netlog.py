from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hawker_agent.browser.cdp import get_cdp

if TYPE_CHECKING:
    from hawker_agent.browser.session import BrowserSession

logger = logging.getLogger(__name__)

NETLOG_INJECT_JS: str = """(function(){
if(window.__netlog_patched) return 'already_patched';
window.__netlog=[];
window.__netlog_patched=true;
var MAX=100, BODY_MAX=5000, REQ_MAX=2000;
function push(e){window.__netlog.push(e);if(window.__netlog.length>MAX)window.__netlog.shift();}
function skip(u){
  var l=u.toLowerCase();
  if(/\\.(png|jpe?g|gif|svg|ico|css|woff2?|ttf|eot|mp[34]|webp|avif)(\\?.*)?$/.test(l)) return true;
  if(l.indexOf('google-analytics.com')!==-1||l.indexOf('googletagmanager.com')!==-1||l.indexOf('sentry.io')!==-1||l.indexOf('hotjar.com')!==-1||l.indexOf('facebook.com/tr')!==-1||l.indexOf('clarity.ms')!==-1||l.indexOf('doubleclick.net')!==-1||l.indexOf('baidu.com/hm.')!==-1) return true;
  if(l.indexOf('/_nuxt/builds/')!==-1||l.indexOf('/__webpack')!==-1||l.indexOf('/hot-update.')!==-1) return true;
  return false;
}
var origFetch=window.fetch;
window.fetch=function(){
  var args=arguments, url=(typeof args[0]==='string')?args[0]:(args[0]&&args[0].url)||'';
  if(skip(url)) return origFetch.apply(this,args);
  var options = args[1] || {};
  var method=(options.method || (args[0] && args[0].method) || 'GET').toUpperCase();
  var reqHeaders = {};
  try {
    var h = options.headers || (args[0] && args[0].headers);
    if (h) {
      if (typeof h.forEach === 'function') {
        h.forEach(function(v, k) { reqHeaders[k] = v; });
      } else {
        reqHeaders = h;
      }
    }
  } catch(e) { reqHeaders = {parse_error: true}; }
  var rb = '';
  try {
    var b = options.body;
    if (b) {
      if (typeof b === 'string') { rb = b; }
      else if (b instanceof FormData || b instanceof URLSearchParams) { rb = b.toString(); }
      else { rb = JSON.stringify(b); }
    }
  } catch(e) { rb = '[parse_error]'; }
  var bodyContent=rb.substring(0,REQ_MAX);
  return origFetch.apply(this,args).then(function(resp){
    var r=resp.clone();
    r.text().then(function(body){
      push({
        url:url, method:method, status:r.status, type:'fetch', 
        headers: reqHeaders, // 捕获请求头
        body:body.substring(0,BODY_MAX), bodyTruncated:body.length>BODY_MAX,
        reqBody:bodyContent, requestBody:bodyContent,
        ts:Date.now()
      });
    }).catch(function(){});
    return resp;
  });
};
var origOpen=XMLHttpRequest.prototype.open, origSend=XMLHttpRequest.prototype.send;
XMLHttpRequest.prototype.open=function(m,u){this._nl_method=m;this._nl_url=u;this._nl_headers={};return origOpen.apply(this,arguments);};
var origSetHeader=XMLHttpRequest.prototype.setRequestHeader;
XMLHttpRequest.prototype.setRequestHeader=function(k,v){if(this._nl_headers)this._nl_headers[k]=v;return origSetHeader.apply(this,arguments);};
XMLHttpRequest.prototype.send=function(){
  var xhr=this;
  var rb = '';
  try {
    var b = arguments[0];
    if (b) {
      if (typeof b === 'string') { rb = b; }
      else if (b instanceof FormData || b instanceof URLSearchParams) { rb = b.toString(); }
      else { rb = JSON.stringify(b); }
    }
  } catch(e) { rb = '[xhr_parse_error]'; }
  var bodyContent = rb.substring(0,REQ_MAX);
  var reqHeaders = xhr._nl_headers || {};
  xhr.addEventListener('load',function(){
    var u=xhr._nl_url||'';
    if(skip(u)) return;
    var body=xhr.responseText||'';
    push({
      url:u, method:(xhr._nl_method||'GET').toUpperCase(), status:xhr.status, type:'xhr', 
      headers: reqHeaders, // 捕获请求头
      body:body.substring(0,BODY_MAX), bodyTruncated:body.length>BODY_MAX,
      reqBody:bodyContent, requestBody:bodyContent,
      ts:Date.now()
    });
  });
  return origSend.apply(this,arguments);
};
return 'patched';
})()"""


async def ensure_network_monitor(session: BrowserSession) -> None:
    """注册网络监听脚本，在每个新页面加载前自动执行（只需调一次）。"""
    if session.netlog_installed:
        return
    try:
        cdp = await get_cdp(session)
        await cdp.cdp_client.send.Page.addScriptToEvaluateOnNewDocument(
            params={"source": NETLOG_INJECT_JS},
            session_id=cdp.session_id,
        )
        session.netlog_installed = True
        logger.debug("网络监听脚本已注册")
    except Exception:
        logger.debug("网络监听脚本注册失败，不影响主流程", exc_info=True)
