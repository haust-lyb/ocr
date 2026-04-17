(function () {
    var ua = navigator.userAgent;
    var isMobile = /Android|iPhone|iPad|iPod|Windows Phone|Mobile/i.test(ua) || window.innerWidth < 768;
    if (!isMobile) return;

    var style = document.createElement('style');
    style.textContent = [
        '#_mobile_mask{',
        '  position:fixed;inset:0;background:#f5f7fa;z-index:99999;',
        '  display:flex;flex-direction:column;align-items:center;justify-content:center;',
        '  text-align:center;padding:40px 30px;',
        '}',
        '#_mobile_mask .icon{font-size:64px;margin-bottom:24px;}',
        '#_mobile_mask h2{color:#0066cc;font-size:1.6em;margin-bottom:16px;',
        '  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}',
        '#_mobile_mask p{color:#666;font-size:1.05em;line-height:1.7;',
        '  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}'
    ].join('');
    document.head.appendChild(style);

    var mask = document.createElement('div');
    mask.id = '_mobile_mask';
    mask.innerHTML = '<div class="icon">\uD83D\uDDA5\uFE0F</div><h2>请在电脑端打开</h2><p>该页面暂不支持手机端使用<br>请使用电脑浏览器访问以获得完整体验</p>';

    if (document.body) {
        document.body.appendChild(mask);
    } else {
        document.addEventListener('DOMContentLoaded', function () {
            document.body.appendChild(mask);
        });
    }
})();
