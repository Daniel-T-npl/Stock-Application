document.addEventListener('DOMContentLoaded', function() {
    // === Strategy to Indicator Mapping ===
    const strategyIndicators = {
        "Trend Following": ["ema_20", "donchian"],
        "Mean Reversion": ["rsi", "bollinger"],
        "Breakout": ["donchian"],
        "Reversal": ["rsi", "bollinger"],
        "Divergence": ["macd", "rsi"],
        "Momentum Breakout": ["ema_20", "donchian"],
        "VWAP Reversion": ["anchored_vwap", "bollinger"],
        "Smart Money Divergence": ["macd", "rsi"],
        "Volatility Expansion Fade": ["bollinger", "keltner"],
        "Trend-Pullback": ["ema_20", "rsi"],
        "Cloud Crossover": ["ichimoku"],
        "Composite Score": ["ema_20", "rsi", "bollinger", "macd"]
    };

    // === Populate Strategy Dropdown ===
    const strategySelect = document.getElementById('strategySelect');
    if (strategySelect) {
        strategySelect.innerHTML = '<option value="">(None)</option>';
        Object.keys(strategyIndicators).forEach(strategy => {
            const opt = document.createElement('option');
            opt.value = strategy;
            opt.textContent = strategy;
            strategySelect.appendChild(opt);
        });
    }

    // Set default date range: last 1 year
    const endDateInput = document.getElementById('endDate');
    const startDateInput = document.getElementById('startDate');
    const today = new Date();
    const lastYear = new Date();
    lastYear.setFullYear(today.getFullYear() - 1);
    endDateInput.value = today.toISOString().slice(0, 10);
    startDateInput.value = lastYear.toISOString().slice(0, 10);

    // Helper to get selected indicators from multiselect
    function getSelectedIndicators() {
        const select = document.getElementById('indicatorSelect');
        return Array.from(select.selectedOptions).map(option => option.value);
    }
    // Helper to get selected symbol from Select2
    function getSelectedSymbol() {
        const select = document.getElementById('symbolSelect');
        return select.value;
    }

    function fetchAndPlot() {
        console.log('fetchAndPlot called');
        const symbol = getSelectedSymbol();
        const indicators = getSelectedIndicators();
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;
        const modifiers = getIndicatorModifiers();
        // Build query string with modifiers as JSON
        const params = new URLSearchParams({
            symbol,
            start: startDate,
            end: endDate,
            indicators: indicators.join(','),
            modifiers: JSON.stringify(modifiers)
        });
        fetch(`/api/ohlcv_indicators/?${params.toString()}`)
            .then(response => response.json())
            .then(json => {
                console.log('API response:', json);
                const data = json.data;
                if (!data || data.length === 0) {
                    Plotly.purge('plotly-chart');
                    Plotly.newPlot('plotly-chart', [{x: [], y: [], type: 'scatter'}], {title: 'No data found'});
                    return;
                }
                const dates = data.map(row => row.date);
                const open = data.map(row => row.open);
                const high = data.map(row => row.high);
                const low = data.map(row => row.low);
                const close = data.map(row => row.close);
                // Candlestick trace: green/red body, green/red wicks, no border
                const candleTrace = {
                    x: dates,
                    open: open,
                    high: high,
                    low: low,
                    close: close,
                    type: 'candlestick',
                    name: 'OHLC',
                    increasing: {
                        line: {color: '#00b894', width: 2}, // green wicks, visible
                        fillcolor: '#00b894' // green body
                    },
                    decreasing: {
                        line: {color: '#d63031', width: 2}, // red wicks, visible
                        fillcolor: '#d63031' // red body
                    },
                    whiskerwidth: 0.5
                };
                // Prepare subplot traces
                let mainTraces = [candleTrace];
                let subplotTraces = [];
                let subplotSpecs = [[{}]];
                let rowHeights = [0.5];
                let subplotCount = 1;
                // EMA
                if (indicators.includes('ema_20')) {
                    const ema20 = data.map(row => row.ema_20);
                    mainTraces.push({
                        x: dates,
                        y: ema20,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'EMA (20)',
                        line: {color: '#ff7f0e', dash: 'dot'},
                        yaxis: 'y1'
                    });
                }
                // Donchian Channel
                if (indicators.includes('donchian')) {
                    const donchian_upper = data.map(row => row.donchian_upper);
                    const donchian_lower = data.map(row => row.donchian_lower);
                    const donchian_avg = donchian_upper.map((v, i) => (v + donchian_lower[i]) / 2);
                    mainTraces.push({
                        x: dates,
                        y: donchian_upper,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Donchian Upper',
                        line: {color: '#00b894', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: donchian_lower,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Donchian Lower',
                        line: {color: '#d63031', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: donchian_avg,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Donchian Avg',
                        line: {color: '#fff', dash: 'dot'},
                        yaxis: 'y1'
                    });
                }
                // Anchored VWAP
                if (indicators.includes('anchored_vwap')) {
                    const vwap = data.map(row => row.anchored_vwap);
                    mainTraces.push({
                        x: dates,
                        y: vwap,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Anchored VWAP',
                        line: {color: '#6c5ce7', dash: 'dot'},
                        yaxis: 'y1'
                    });
                }
                // Bollinger Bands
                if (indicators.includes('bollinger')) {
                    const ma = data.map(row => row.ma);
                    const bb_upper = data.map(row => row.bb_upper);
                    const bb_lower = data.map(row => row.bb_lower);
                    mainTraces.push({
                        x: dates,
                        y: ma,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MA (Bollinger)',
                        line: {color: '#f1c40f', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: bb_upper,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'BB Upper',
                        line: {color: '#e17055', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: bb_lower,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'BB Lower',
                        line: {color: '#00b894', dash: 'dot'},
                        yaxis: 'y1'
                    });
                }
                // Ichimoku Cloud
                if (indicators.includes('ichimoku')) {
                    const tenkan = data.map(row => row.tenkan_sen);
                    const kijun = data.map(row => row.kijun_sen);
                    const spanA = data.map(row => row.senkou_span_a);
                    const spanB = data.map(row => row.senkou_span_b);
                    const chikou = data.map(row => row.chikou_span);
                    mainTraces.push({x: dates, y: tenkan, type: 'scatter', mode: 'lines', name: 'Tenkan-sen', line: {color: '#e17055'}, yaxis: 'y1'});
                    mainTraces.push({x: dates, y: kijun, type: 'scatter', mode: 'lines', name: 'Kijun-sen', line: {color: '#0984e3'}, yaxis: 'y1'});
                    mainTraces.push({x: dates, y: spanA, type: 'scatter', mode: 'lines', name: 'Senkou Span A', line: {color: '#00b894', dash: 'dot'}, yaxis: 'y1'});
                    mainTraces.push({x: dates, y: spanB, type: 'scatter', mode: 'lines', name: 'Senkou Span B', line: {color: '#fdcb6e', dash: 'dot'}, yaxis: 'y1'});
                    mainTraces.push({x: dates, y: chikou, type: 'scatter', mode: 'lines', name: 'Chikou Span', line: {color: '#636e72'}, yaxis: 'y1'});
                }
                // Fibonacci Retracement
                if (indicators.includes('fibonacci')) {
                    const fibLevels = ['fibonacci_0', 'fibonacci_23', 'fibonacci_38', 'fibonacci_50', 'fibonacci_61', 'fibonacci_78', 'fibonacci_100'];
                    fibLevels.forEach((level, i) => {
                        if (data[0] && data[0][level] !== undefined) {
                            mainTraces.push({
                                x: dates,
                                y: data.map(row => row[level]),
                                type: 'scatter',
                                mode: 'lines',
                                name: `Fib ${level.split('_')[1]}%`,
                                line: {color: '#b2bec3', dash: 'dot'},
                                yaxis: 'y1'
                            });
                        }
                    });
                }
                // Keltner Channels (main chart)
                if (indicators.includes('keltner')) {
                    const kc_middle = data.map(row => row.kc_middle);
                    const kc_upper = data.map(row => row.kc_upper);
                    const kc_lower = data.map(row => row.kc_lower);
                    mainTraces.push({
                        x: dates,
                        y: kc_middle,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Keltner Middle',
                        line: {color: '#00b894', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: kc_upper,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Keltner Upper',
                        line: {color: '#fdcb6e', dash: 'dot'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: kc_lower,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Keltner Lower',
                        line: {color: '#e17055', dash: 'dot'},
                        yaxis: 'y1'
                    });
                }
                // Parabolic SAR (main chart, as dots)
                if (indicators.includes('parabolic_sar')) {
                    const psar = data.map(row => row.parabolic_sar);
                    mainTraces.push({
                        x: dates,
                        y: psar,
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Parabolic SAR',
                        marker: {color: '#6c5ce7', size: 6, symbol: 'circle-open'},
                        yaxis: 'y1'
                    });
                }
                // Prepare subplots for each indicator
                function addSubplot(traces, title, height=0.2) {
                    if (traces.length > 0) {
                        subplotTraces.push(traces);
                        subplotSpecs.push([{}]);
                        rowHeights.push(height);
                        subplotCount++;
                    }
                }
                // MACD
                let macdTraces = [];
                if (indicators.includes('macd')) {
                    const macd = data.map(row => row.macd);
                    const macd_signal = data.map(row => row.macd_signal);
                    const macd_hist = data.map(row => row.macd_hist);
                    macdTraces.push({
                        x: dates,
                        y: macd,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MACD',
                        line: {color: '#2ca02c'},
                        yaxis: `y${subplotCount+1}`
                    });
                    macdTraces.push({
                        x: dates,
                        y: macd_signal,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MACD Signal',
                        line: {color: '#d62728', dash: 'dot'},
                        yaxis: `y${subplotCount+1}`
                    });
                    macdTraces.push({
                        x: dates,
                        y: macd_hist,
                        type: 'bar',
                        name: 'MACD Hist',
                        marker: {color: '#9467bd'},
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(macdTraces, 'MACD');
                // Stochastic Oscillator
                let stochTraces = [];
                if (indicators.includes('stoch')) {
                    const stoch_k = data.map(row => row.stoch_k);
                    const stoch_d = data.map(row => row.stoch_d);
                    stochTraces.push({
                        x: dates,
                        y: stoch_k,
                        type: 'scatter',
                        mode: 'lines',
                        name: '%K (Stoch)',
                        line: {color: '#0984e3'},
                        yaxis: `y${subplotCount+1}`
                    });
                    stochTraces.push({
                        x: dates,
                        y: stoch_d,
                        type: 'scatter',
                        mode: 'lines',
                        name: '%D (Stoch)',
                        line: {color: '#fdcb6e', dash: 'dot'},
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(stochTraces, 'Stochastic');
                // RSI
                let rsiTraces = [];
                if (indicators.includes('rsi')) {
                    const rsi = data.map(row => row.rsi_14);
                    rsiTraces.push({
                        x: dates,
                        y: rsi,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'RSI (14)',
                        line: {color: '#6c5ce7'},
                        yaxis: `y${subplotCount+1}`
                    });
                    // Add overbought/oversold lines
                    const overbought = parseInt(document.getElementById('rsi_overbought').value);
                    const oversold = parseInt(document.getElementById('rsi_oversold').value);
                    rsiTraces.push({
                        x: [dates[0], dates[dates.length-1]],
                        y: [overbought, overbought],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Overbought',
                        line: {color: '#e17055', dash: 'dash'},
                        yaxis: `y${subplotCount+1}`,
                        showlegend: true
                    });
                    rsiTraces.push({
                        x: [dates[0], dates[dates.length-1]],
                        y: [oversold, oversold],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Oversold',
                        line: {color: '#00b894', dash: 'dash'},
                        yaxis: `y${subplotCount+1}`,
                        showlegend: true
                    });
                }
                addSubplot(rsiTraces, 'RSI');
                // ADX (subplot)
                let adxTraces = [];
                if (indicators.includes('adx')) {
                    const adx = data.map(row => row.adx);
                    adxTraces.push({
                        x: dates,
                        y: adx,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'ADX',
                        line: {color: '#00b894'},
                        yaxis: `y${subplotCount+1}`
                    });
                    // Add 25/20 reference lines
                    adxTraces.push({
                        x: [dates[0], dates[dates.length-1]],
                        y: [25, 25],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Strong Trend (25)',
                        line: {color: '#fdcb6e', dash: 'dash'},
                        yaxis: `y${subplotCount+1}`,
                        showlegend: true
                    });
                    adxTraces.push({
                        x: [dates[0], dates[dates.length-1]],
                        y: [20, 20],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Range Bound (20)',
                        line: {color: '#636e72', dash: 'dot'},
                        yaxis: `y${subplotCount+1}`,
                        showlegend: true
                    });
                }
                addSubplot(adxTraces, 'ADX');
                // OBV (subplot)
                let obvTraces = [];
                if (indicators.includes('obv')) {
                    const obv = data.map(row => row.obv);
                    obvTraces.push({
                        x: dates,
                        y: obv,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'OBV',
                        line: {color: '#0984e3'},
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(obvTraces, 'OBV');
                // VPT (subplot)
                let vptTraces = [];
                if (indicators.includes('vpt')) {
                    const vpt = data.map(row => row.vpt);
                    vptTraces.push({
                        x: dates,
                        y: vpt,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'VPT',
                        line: {color: '#fdcb6e'},
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(vptTraces, 'VPT');
                // CMF (subplot, fill region green above 0, red below 0)
                let cmfTraces = [];
                if (indicators.includes('cmf')) {
                    const cmf = data.map(row => row.cmf);
                    // Split into positive and negative for fill color
                    let cmf_pos = cmf.map(v => v > 0 ? v : 0);
                    let cmf_neg = cmf.map(v => v < 0 ? v : 0);
                    cmfTraces.push({
                        x: dates,
                        y: cmf_pos,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'CMF+',
                        line: {color: '#00b894'},
                        fill: 'tozeroy',
                        fillcolor: 'rgba(0,184,148,0.3)',
                        yaxis: `y${subplotCount+1}`
                    });
                    cmfTraces.push({
                        x: dates,
                        y: cmf_neg,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'CMF-',
                        line: {color: '#d63031'},
                        fill: 'tozeroy',
                        fillcolor: 'rgba(214,48,49,0.3)',
                        yaxis: `y${subplotCount+1}`
                    });
                    // Add a zero line for reference
                    cmfTraces.push({
                        x: [dates[0], dates[dates.length-1]],
                        y: [0, 0],
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Zero',
                        line: {color: '#fff', dash: 'dot'},
                        yaxis: `y${subplotCount+1}`,
                        showlegend: false
                    });
                }
                addSubplot(cmfTraces, 'CMF');
                // ATR (subplot)
                let atrTraces = [];
                if (indicators.includes('atr')) {
                    const atr = data.map(row => row.atr);
                    atrTraces.push({
                        x: dates,
                        y: atr,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'ATR',
                        line: {color: '#6c5ce7'},
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(atrTraces, 'ATR');
                // Fractals (main chart, show as markers for highs/lows)
                if (indicators.includes('fractals')) {
                    const fractal_high = data.map(row => row.fractal_high);
                    const fractal_low = data.map(row => row.fractal_low);
                    mainTraces.push({
                        x: dates,
                        y: fractal_high,
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Fractal High',
                        marker: {color: '#e17055', size: 8, symbol: 'triangle-up'},
                        yaxis: 'y1'
                    });
                    mainTraces.push({
                        x: dates,
                        y: fractal_low,
                        type: 'scatter',
                        mode: 'markers',
                        name: 'Fractal Low',
                        marker: {color: '#00b894', size: 8, symbol: 'triangle-down'},
                        yaxis: 'y1'
                    });
                }
                // Volume Profile (as horizontal bar chart in its own subplot)
                let vpTraces = [];
                console.log('DEBUG: json.volume_profile =', json.volume_profile);
                if (indicators.includes('volume_profile') && json.volume_profile && json.volume_profile.length > 0) {
                    const vp = json.volume_profile;
                    console.log('Raw Volume Profile:', vp);
                    let priceBins = vp.map(row => {
                        if (!row.price_bin) return null;
                        const match = row.price_bin.match(/[\[(](.*),\s*(.*)[\])]/);
                        if (match) {
                            const low = parseFloat(match[1]);
                            const high = parseFloat(match[2]);
                            if (low === high) return low;
                            return (low + high) / 2;
                        }
                        return null;
                    });
                    let filtered = vp.map((row, i) => ({...row, price: priceBins[i]})).filter(row => row.price !== null && !isNaN(row.price));
                    if (filtered.length === 0 && vp.length === 1 && vp[0].price_bin) {
                        const match = vp[0].price_bin.match(/\[(.*),(.*)\]/);
                        if (match) {
                            const low = parseFloat(match[1]);
                            const high = parseFloat(match[2]);
                            const center = (low + high) / 2;
                            filtered = [{price: center, volume: vp[0].volume}];
                        }
                    }
                    if (filtered.length === 0) {
                        console.warn('No valid price bins for volume profile:', vp);
                    }
                    let sorted = filtered.sort((a, b) => a.price - b.price);
                    priceBins = sorted.map(row => row.price);
                    const volumes = sorted.map(row => row.volume);
                    vpTraces.push({
                        x: volumes,
                        y: priceBins,
                        type: 'bar',
                        orientation: 'h',
                        name: 'Volume Profile',
                        marker: {color: 'rgba(95,123,232,0.7)'},
                        opacity: 0.7,
                        width: 8,
                        showlegend: false,
                        yaxis: `y${subplotCount+1}`
                    });
                }
                addSubplot(vpTraces, 'Volume Profile', 0.25);
                // Combine all traces for subplots
                let allTraces = [...mainTraces];
                let layout = {
                    title: `${symbol} Price & Indicators`,
                    plot_bgcolor: '#18191a',
                    paper_bgcolor: '#18191a',
                    font: {color: '#fff'},
                    grid: {rows: subplotSpecs.length, columns: 1, roworder: 'top to bottom'},
                    height: 900 + 200 * (subplotSpecs.length - 1),
                    margin: {t: 50, b: 80},
                    legend: {orientation: 'h', y: -0.2},
                    shapes: Array.from({length: subplotSpecs.length}, (_, i) => ({
                        type: 'rect',
                        xref: `paper`,
                        yref: `paper`,
                        x0: 0,
                        x1: 1,
                        y0: 1 - rowHeights.slice(0, i+1).reduce((a, b) => a + b, 0),
                        y1: 1 - rowHeights.slice(0, i).reduce((a, b) => a + b, 0),
                        line: {color: '#fff', width: 2},
                        layer: 'below',
                        opacity: 0.2,
                        fillcolor: 'rgba(0,0,0,0)'
                    }))
                };
                // Set up axes for each subplot
                layout['xaxis'] = {title: '', rangeslider: {visible: false}, color: '#fff', gridcolor: '#333', domain: [0, 1], anchor: 'y1', showticklabels: false};
                layout['yaxis'] = {title: 'Price', color: '#fff', gridcolor: '#333', domain: [1 - rowHeights[0], 1]};
                // Show date labels if only the candlestick is present (no indicators, only one subplot)
                if (indicators.length === 0 && subplotSpecs.length === 1) {
                    layout['xaxis'].showticklabels = true;
                    layout['xaxis'].title = 'Date';
                }
                for (let i = 1, yStart = 1 - rowHeights[0]; i < subplotSpecs.length; i++) {
                    const yEnd = yStart;
                    yStart -= rowHeights[i];
                    layout[`xaxis${i+1}`] = {title: i === subplotSpecs.length - 1 ? 'Date' : '', color: '#fff', gridcolor: '#333', domain: [0, 1], anchor: `y${i+1}`, showticklabels: i === subplotSpecs.length - 1};
                    layout[`yaxis${i+1}`] = {title: subplotSpecs[i][0] && subplotSpecs[i][0].title ? subplotSpecs[i][0].title : '', color: '#fff', gridcolor: '#333', domain: [yStart, yEnd]};
                }
                // Add all subplot traces
                subplotTraces.forEach((traces, i) => {
                    allTraces = allTraces.concat(traces.map(t => ({...t, xaxis: `x${i+2}`, yaxis: `y${i+2}`})));
                });
                Plotly.purge('plotly-chart');
                console.log('Plotly traces:', allTraces);
                Plotly.newPlot('plotly-chart', allTraces, layout, {responsive: true});
                // Add debug log for volume profile
                console.log('Volume Profile data:', json.volume_profile);
                if (indicators.includes('volume_profile') && (!json.volume_profile || json.volume_profile.length === 0)) {
                    const warning = document.getElementById('volumeProfileWarning');
                    if (warning) warning.style.display = 'block';
                } else {
                    const warning = document.getElementById('volumeProfileWarning');
                    if (warning) warning.style.display = 'none';
                }
                // At the end of fetchAndPlot, call fetchAndOverlaySignals
                fetchAndOverlaySignals();
            });
    }
    document.getElementById('updateChart').addEventListener('click', fetchAndPlot);
    // Also update on date, symbol, or indicator change
    startDateInput.addEventListener('change', fetchAndPlot);
    endDateInput.addEventListener('change', fetchAndPlot);
    $('#symbolSelect').on('change', fetchAndPlot);
    $('#indicatorSelect').on('change', fetchAndPlot);
    // Initial plot (no indicators)
    fetchAndPlot();
    // Collect indicator modifier values from UI
    function getIndicatorModifiers() {
        const modifiers = {};
        // Bollinger
        modifiers.bollinger = {
            ma: parseInt(document.getElementById('bollinger_ma').value),
            stddev: parseFloat(document.getElementById('bollinger_stddev').value)
        };
        // RSI
        modifiers.rsi = {
            period: parseInt(document.getElementById('rsi_period').value),
            overbought: parseInt(document.getElementById('rsi_overbought').value),
            oversold: parseInt(document.getElementById('rsi_oversold').value)
        };
        // EMA
        modifiers.ema_20 = {
            ma: parseInt(document.getElementById('ema_ma').value)
        };
        // Anchored VWAP
        modifiers.anchored_vwap = {
            start_date: document.getElementById('anchored_vwap_start').value
        };
        // Fibonacci
        modifiers.fibonacci = {
            pos0: document.getElementById('fib_pos0').value,
            pos1: document.getElementById('fib_pos1').value
        };
        // Stochastic, Donchian, Volume Profile, Ichimoku: no modifiers for now
        return modifiers;
    }
    // When fetching, send modifiers
    const modifiers = getIndicatorModifiers();
    // Add to fetch URL or POST body as needed
    // Show/hide indicator modifier controls based on selection
    function updateModifierVisibility() {
        const selected = $("#indicatorSelect").val() || [];
        // Bollinger
        document.getElementById('bollinger_modifiers').style.display = selected.includes('bollinger') ? '' : 'none';
        // RSI
        document.getElementById('rsi_modifiers').style.display = selected.includes('rsi') ? '' : 'none';
        // EMA
        document.getElementById('ema_modifiers').style.display = selected.includes('ema_20') ? '' : 'none';
        // Anchored VWAP
        document.getElementById('anchored_vwap_modifiers').style.display = selected.includes('anchored_vwap') ? '' : 'none';
        // Fibonacci
        document.getElementById('fibonacci_modifiers').style.display = selected.includes('fibonacci') ? '' : 'none';
    }
    $('#indicatorSelect').on('change', updateModifierVisibility);
    document.addEventListener('DOMContentLoaded', updateModifierVisibility);
    // Set Anchored VWAP start date to match chart start date by default
    document.getElementById('startDate').addEventListener('change', function() {
        document.getElementById('anchored_vwap_start').value = this.value;
    });
    // Also update chart when Anchored VWAP start date changes
    document.getElementById('anchored_vwap_start').addEventListener('change', fetchAndPlot);
    // === Strategy Selection Logic ===
    if (strategySelect) {
        strategySelect.addEventListener('change', function() {
            const selectedStrategy = strategySelect.value;
            const indicatorSelect = document.getElementById('indicatorSelect');
            if (!selectedStrategy || !strategyIndicators[selectedStrategy]) {
                // If (None) selected, do not change indicators
                fetchAndOverlaySignals();
                return;
            }
            const indicators = strategyIndicators[selectedStrategy];
            // Deselect all
            Array.from(indicatorSelect.options).forEach(opt => { opt.selected = false; });
            // Select only those needed
            Array.from(indicatorSelect.options).forEach(opt => {
                if (indicators.includes(opt.value)) {
                    opt.selected = true;
                }
            });
            // If using Select2 or similar, trigger update
            if (window.jQuery && $(indicatorSelect).data('select2')) {
                $(indicatorSelect).trigger('change.select2');
            } else {
                // Otherwise, trigger change event for vanilla JS
                indicatorSelect.dispatchEvent(new Event('change'));
            }
            // After updating indicators, call fetchAndOverlaySignals
            fetchAndOverlaySignals();
        });
    }
    // === Fetch and Overlay Buy/Sell Signals ===
    function fetchAndOverlaySignals() {
        console.log('fetchAndOverlaySignals called');
        const selectedStrategy = strategySelect ? strategySelect.value : null;
        const symbol = getSelectedSymbol();
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;
        if (!selectedStrategy) {
            // No strategy selected, remove any existing signal overlays
            removeSignalMarkers();
            return;
        }
        const url = `/api/strategy_signals/?symbol=${symbol}&strategy=${encodeURIComponent(selectedStrategy)}&start=${startDate}&end=${endDate}`;
        console.log('Fetching strategy signals from:', url);
        fetch(url)
            .then(response => response.json())
            .then(json => {
                if (!json.signals) {
                    removeSignalMarkers();
                    return;
                }
                overlaySignalMarkers(json.signals);
            })
            .catch(err => {
                console.error('Error fetching strategy signals:', err);
                removeSignalMarkers();
            });
    }

    // === Overlay Signal Markers on Plotly Chart ===
    function overlaySignalMarkers(signals) {
        // Remove previous signal traces
        removeSignalMarkers();
        if (!signals || signals.length === 0) return;
        const buySignals = signals.filter(s => s.type === 'buy');
        const sellSignals = signals.filter(s => s.type === 'sell');
        // Add buy markers (green up arrow)
        const buyTrace = {
            x: buySignals.map(s => s.date),
            y: buySignals.map(s => s.price),
            mode: 'markers',
            name: 'Buy',
            marker: {
                symbol: 'arrow-up',
                color: 'lime',
                size: 20,
                line: { color: 'white', width: 3 }
            },
            type: 'scatter',
            yaxis: 'y1',
            customdata: buySignals,
            hovertemplate: 'Buy: %{y} on %{x}<extra></extra>',
            showlegend: true,
            uid: 'buy-signal-trace'
        };
        // Add sell markers (red down arrow)
        const sellTrace = {
            x: sellSignals.map(s => s.date),
            y: sellSignals.map(s => s.price),
            mode: 'markers',
            name: 'Sell',
            marker: {
                symbol: 'arrow-down',
                color: 'red',
                size: 20,
                line: { color: 'white', width: 3 }
            },
            type: 'scatter',
            yaxis: 'y1',
            customdata: sellSignals,
            hovertemplate: 'Sell: %{y} on %{x}<extra></extra>',
            showlegend: true,
            uid: 'sell-signal-trace'
        };
        // Add to chart
        Plotly.addTraces('plotly-chart', [buyTrace, sellTrace]);
    }

    // === Remove Signal Markers from Plotly Chart ===
    function removeSignalMarkers() {
        const chartDiv = document.getElementById('plotly-chart');
        if (!chartDiv || !chartDiv.data) return;
        // Remove traces with our custom uids
        const toRemove = [];
        chartDiv.data.forEach((trace, idx) => {
            if (trace.uid === 'buy-signal-trace' || trace.uid === 'sell-signal-trace') {
                toRemove.push(idx);
            }
        });
        if (toRemove.length > 0) {
            Plotly.deleteTraces('plotly-chart', toRemove);
        }
    }
}); 