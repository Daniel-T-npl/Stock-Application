document.addEventListener('DOMContentLoaded', function() {
    // Set default date range: last 1 year
    const endDateInput = document.getElementById('endDate');
    const startDateInput = document.getElementById('startDate');
    const today = new Date();
    const lastYear = new Date();
    lastYear.setFullYear(today.getFullYear() - 1);
    endDateInput.value = today.toISOString().slice(0, 10);
    startDateInput.value = lastYear.toISOString().slice(0, 10);

    // Helper to get selected indicators from checkboxes
    function getSelectedIndicators() {
        return Array.from(document.querySelectorAll('.indicator-checkbox:checked')).map(cb => cb.value);
    }

    // Keep dropdown open and allow checkbox selection
    document.querySelectorAll('#indicatorDropdownMenu label').forEach(label => {
        label.addEventListener('mousedown', function(e) {
            e.preventDefault(); // Prevent dropdown from closing
            const cb = label.querySelector('input[type="checkbox"]');
            cb.checked = !cb.checked;
            cb.dispatchEvent(new Event('change', {bubbles: true}));
        });
    });

    function fetchAndPlot() {
        const symbol = document.getElementById('symbolSelect').value;
        const start = document.getElementById('startDate').value;
        const end = document.getElementById('endDate').value;
        const selectedIndicators = getSelectedIndicators();
        let url = `/api/ohlcv_indicators/?symbol=${symbol}`;
        if (start) url += `&start=${start}`;
        if (end) url += `&end=${end}`;
        fetch(url)
            .then(response => response.json())
            .then(json => {
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
                // Candlestick trace: green/red body, white thin wicks, no border
                const candleTrace = {
                    x: dates,
                    open: open,
                    high: high,
                    low: low,
                    close: close,
                    type: 'candlestick',
                    name: 'OHLC',
                    increasing: {
                        line: {color: 'white', width: 1}, // wicks
                        fillcolor: '#00b894' // green
                    },
                    decreasing: {
                        line: {color: 'white', width: 1}, // wicks
                        fillcolor: '#d63031' // red
                    },
                    whiskerwidth: 0.5
                };
                let traces = [candleTrace];
                // EMA
                if (selectedIndicators.includes('ema_20')) {
                    const ema20 = data.map(row => row.ema_20);
                    traces.push({
                        x: dates,
                        y: ema20,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'EMA (20)',
                        line: {color: '#ff7f0e', dash: 'dot'}
                    });
                }
                // MACD
                if (selectedIndicators.includes('macd')) {
                    const macd = data.map(row => row.macd);
                    const macd_signal = data.map(row => row.macd_signal);
                    const macd_hist = data.map(row => row.macd_hist);
                    traces.push({
                        x: dates,
                        y: macd,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MACD',
                        line: {color: '#2ca02c'},
                        yaxis: 'y2'
                    });
                    traces.push({
                        x: dates,
                        y: macd_signal,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'MACD Signal',
                        line: {color: '#d62728', dash: 'dot'},
                        yaxis: 'y2'
                    });
                    traces.push({
                        x: dates,
                        y: macd_hist,
                        type: 'bar',
                        name: 'MACD Hist',
                        marker: {color: '#9467bd'},
                        yaxis: 'y2'
                    });
                }
                // Stochastic Oscillator
                if (selectedIndicators.includes('stoch')) {
                    const stoch_k = data.map(row => row.stoch_k);
                    const stoch_d = data.map(row => row.stoch_d);
                    traces.push({
                        x: dates,
                        y: stoch_k,
                        type: 'scatter',
                        mode: 'lines',
                        name: '%K (Stoch)',
                        line: {color: '#0984e3'},
                        yaxis: 'y3'
                    });
                    traces.push({
                        x: dates,
                        y: stoch_d,
                        type: 'scatter',
                        mode: 'lines',
                        name: '%D (Stoch)',
                        line: {color: '#fdcb6e', dash: 'dot'},
                        yaxis: 'y3'
                    });
                }
                // Donchian Channel
                if (selectedIndicators.includes('donchian')) {
                    const donchian_upper = data.map(row => row.donchian_upper);
                    const donchian_lower = data.map(row => row.donchian_lower);
                    traces.push({
                        x: dates,
                        y: donchian_upper,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Donchian Upper',
                        line: {color: '#00b894', dash: 'dot'}
                    });
                    traces.push({
                        x: dates,
                        y: donchian_lower,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Donchian Lower',
                        line: {color: '#d63031', dash: 'dot'}
                    });
                }
                // Anchored VWAP
                if (selectedIndicators.includes('anchored_vwap')) {
                    const vwap = data.map(row => row.anchored_vwap);
                    traces.push({
                        x: dates,
                        y: vwap,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Anchored VWAP',
                        line: {color: '#6c5ce7', dash: 'dot'}
                    });
                }
                // Layout with dark background and white axes
                const layout = {
                    title: `${symbol} Price & Indicators`,
                    plot_bgcolor: '#18191a',
                    paper_bgcolor: '#18191a',
                    font: {color: '#fff'},
                    xaxis: {title: 'Date', rangeslider: {visible: false}, color: '#fff', gridcolor: '#333'},
                    yaxis: {title: 'Price', color: '#fff', gridcolor: '#333'},
                    yaxis2: {
                        title: 'MACD',
                        overlaying: 'y',
                        side: 'right',
                        showgrid: false,
                        zeroline: true,
                        color: '#fff'
                    },
                    yaxis3: {
                        title: 'Stochastic',
                        overlaying: 'y',
                        side: 'left',
                        position: 0.05,
                        anchor: 'x',
                        showgrid: false,
                        zeroline: true,
                        color: '#fff'
                    },
                    legend: {orientation: 'h', y: -0.2},
                    height: 600,
                    margin: {t: 50, b: 80}
                };
                Plotly.newPlot('plotly-chart', traces, layout, {responsive: true});
            });
    }
    document.getElementById('updateChart').addEventListener('click', fetchAndPlot);
    // Also update on indicator checkbox change
    document.querySelectorAll('.indicator-checkbox').forEach(cb => {
        cb.addEventListener('change', fetchAndPlot);
    });
    // Initial plot (no indicators)
    fetchAndPlot();
}); 