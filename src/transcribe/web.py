"""HTML/CSS/JS frontend template for Transcribe app with multi-model comparison and benchmarks."""

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcribe - AI Speech, Diarization & Model Comparison</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = { darkMode: 'class' };
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  </script>
</head>
<body class="bg-slate-50 text-slate-800 dark:bg-slate-950 dark:text-slate-100 min-h-screen flex flex-col font-sans antialiased transition-colors duration-200">
  <div id="toast" class="fixed top-4 right-4 z-50 transform transition-all duration-300 translate-y-[-100px] opacity-0 bg-indigo-600 text-white text-xs font-semibold px-4 py-2 rounded-xl shadow-xl flex items-center gap-2">
    <span>✔</span> <span id="toast-msg">Copied!</span>
  </div>

  <header class="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/50 backdrop-blur sticky top-0 z-40 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">TR</div>
      <div>
        <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent">Transcribe</h1>
      </div>
    </div>
    <div class="flex items-center gap-2 text-xs">
      <button id="btn-open-compare" class="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-50 hover:bg-indigo-100 dark:bg-indigo-950/60 dark:hover:bg-indigo-900/60 text-indigo-700 dark:text-indigo-300 rounded-lg border border-indigo-200 dark:border-indigo-800/80 font-semibold transition-colors">
        <span>⚖️ Compare Models</span>
      </button>
      <button id="btn-open-history" class="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg border border-slate-300 dark:border-slate-700 font-medium">
        <svg class="w-3.5 h-3.5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        History
      </button>
      <button id="btn-token-modal" class="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-lg border border-slate-300 dark:border-slate-700 font-medium">
        <span id="token-label">🔑 Set Token</span>
      </button>
      <button id="btn-theme-toggle" class="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-300 dark:border-slate-700 transition-colors">
        <svg id="theme-sun" class="w-4 h-4 hidden dark:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/></svg>
        <svg id="theme-moon" class="w-4 h-4 block dark:hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/></svg>
      </button>
    </div>
  </header>

  <main class="flex-1 max-w-5xl w-full mx-auto p-6 space-y-6">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="md:col-span-1 space-y-6">
        <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
          <h2 class="font-semibold text-slate-800 dark:text-slate-200 text-sm tracking-wider uppercase">Configuration</h2>
          <!-- Cascaded Model Selectors -->
          <div class="space-y-3">
            <div>
              <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Architecture / Family</label>
              <select id="family-select" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 font-semibold"></select>
            </div>

            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Variant / Size</label>
                <select id="variant-select" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 font-medium"></select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Compute Type</label>
                <select id="compute-type-select" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 font-medium">
                  <option value="default" selected>Auto / Default</option>
                  <option value="float16">FP16 (Fast GPU)</option>
                  <option value="int8">INT8 (Low VRAM)</option>
                  <option value="int8_float16">INT8_FP16 (Hybrid)</option>
                </select>
              </div>
            </div>

            <input type="hidden" id="model-select" value="base" />

            <!-- Real-time Badges -->
            <div id="model-badges" class="flex flex-wrap items-center gap-1 mt-1 text-[10px]">
              <span id="badge-cache" class="px-1.5 py-0.5 rounded font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">● Local</span>
              <span id="badge-params" class="px-1.5 py-0.5 rounded font-medium bg-slate-500/10 text-slate-600 dark:text-slate-400 border border-slate-500/20">74M params</span>
              <span id="badge-vram" class="px-1.5 py-0.5 rounded font-medium bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20">~1 GB VRAM</span>
              <span id="badge-speed" class="px-1.5 py-0.5 rounded font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">~16x RTF</span>
            </div>
          </div>

          <!-- Dynamic Adaptive Knobs Panel -->
          <div id="adaptive-knobs" class="p-3 bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800/80 rounded-xl space-y-2 text-xs">
            <!-- Dynamically mounted based on active model family -->
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">Language</label>
            <input type="text" id="lang-input" placeholder="Auto-detect (e.g. en, id, es)" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500">
          </div>
          <div class="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500">
            <span>Speaker Diarization</span>
            <span class="inline-flex items-center text-indigo-600 dark:text-indigo-400 font-medium">✓ Auto-Enabled</span>
          </div>
        </div>

        <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 shadow-sm">
          <div class="flex items-center justify-between mb-3 border-b border-slate-200 dark:border-slate-800 pb-2">
            <button id="tab-file" class="text-xs font-semibold pb-1 border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400">📁 File Upload</button>
            <button id="tab-url" class="text-xs font-semibold pb-1 border-b-2 border-transparent text-slate-500 dark:text-slate-400">🔗 Drive / URL</button>
          </div>
          <div id="panel-file" class="space-y-2">
            <div id="dropzone" class="border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-indigo-500 transition-colors rounded-xl p-5 text-center cursor-pointer bg-slate-50/50 dark:bg-slate-950/40">
              <svg class="mx-auto h-8 w-8 text-slate-400 dark:text-slate-500 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
              <p class="text-xs text-slate-600 dark:text-slate-400"><span class="text-indigo-600 dark:text-indigo-400 font-semibold">Upload file</span> or drag</p>
              <input type="file" id="file-input" class="hidden" accept="audio/*,video/*">
            </div>
            <div id="file-info" class="text-xs text-slate-700 dark:text-slate-300 font-medium truncate hidden"></div>
          </div>
          <div id="panel-url" class="space-y-2 hidden">
            <input type="url" id="url-input" placeholder="https://drive.google.com/file/d/.../view" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500">
          </div>
          <div class="flex items-center gap-2 mt-3 pt-2 border-t border-slate-200 dark:border-slate-800">
            <input type="checkbox" id="chk-force" class="rounded text-indigo-600 focus:ring-indigo-500 border-slate-300 dark:border-slate-700 cursor-pointer">
            <label for="chk-force" class="text-[11px] font-medium text-slate-600 dark:text-slate-400 cursor-pointer">⚡ Force fresh run (bypass cache)</label>
          </div>
          <button id="btn-transcribe" disabled class="w-full mt-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium py-2.5 px-4 rounded-xl text-sm transition-all shadow-lg shadow-indigo-600/20">
            Start Transcription
          </button>
        </div>
      </div>

      <div class="md:col-span-2 space-y-6">
        <div class="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 min-h-[460px] flex flex-col shadow-sm">
          <div class="flex items-center justify-between pb-4 border-b border-slate-200 dark:border-slate-800 mb-3">
            <div>
              <h2 class="font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                Transcription Timeline
                <span id="stream-badge" class="hidden text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 animate-pulse">Live</span>
              </h2>
              <div class="flex items-center gap-2 mt-0.5">
                <p id="meta-label" class="text-xs text-slate-500 dark:text-slate-400">Ready to transcribe.</p>
                <button id="btn-resume-action" onclick="resumeTranscription()" class="hidden px-2.5 py-0.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[11px] font-semibold flex items-center gap-1 shadow-sm transition-all animate-pulse">
                  ▶️ Resume
                </button>
              </div>
            </div>
            <div id="export-actions" class="flex flex-wrap items-center gap-1.5 opacity-30 pointer-events-none transition-opacity">
              <select id="export-style" class="bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs rounded-lg px-2 py-1 focus:outline-none focus:border-indigo-500 font-medium">
                <option value="full">Full (Time + Speaker)</option>
                <option value="no_ts">Without Timestamps</option>
                <option value="speaker_only">With Speaker Only</option>
                <option value="text_only">Only Text</option>
              </select>
              <!-- MOM Button (Hidden pending refined implementation) -->
              <button id="btn-mom" onclick="openMomModal()" class="hidden px-2 py-1 bg-amber-600 hover:bg-amber-500 text-xs text-white rounded-lg font-bold items-center gap-1 shadow-sm transition-all" title="Generate Minutes of Meeting">📝 MOM</button>
              <button onclick="openRefineModal()" class="px-2 py-1 bg-teal-600 hover:bg-teal-500 text-xs text-white rounded-lg font-bold flex items-center gap-1 shadow-sm transition-all" title="Refine and polish transcript grammar, punctuation & disfluencies">✨ Refine</button>
              <button onclick="copyMarkdown()" class="px-2 py-1 bg-indigo-600 hover:bg-indigo-500 text-xs text-white rounded-lg font-medium flex items-center gap-1 shadow-sm">📋 Copy</button>
              <button onclick="downloadFormat('md')" class="px-2 py-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-xs rounded-lg text-slate-700 dark:text-slate-300 font-medium">⬇️ MD</button>
              <button onclick="downloadFormat('txt')" class="px-2 py-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-xs rounded-lg text-slate-700 dark:text-slate-300">TXT</button>
              <button onclick="downloadFormat('srt')" class="px-2 py-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-xs rounded-lg text-slate-700 dark:text-slate-300">SRT</button>
              <button onclick="downloadFormat('vtt')" class="px-2 py-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-xs rounded-lg text-slate-700 dark:text-slate-300">VTT</button>
              <button onclick="downloadFormat('json')" class="px-2 py-1 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-xs rounded-lg text-slate-700 dark:text-slate-300">JSON</button>
              <button onclick="clearCurrentTimeline()" class="px-2 py-1 bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-400 text-xs rounded-lg border border-rose-500/20 font-medium" title="Clear current timeline">🧹 Clear</button>
            </div>
          </div>

          <!-- COMPARISON & MULTI-MODEL QUICK RUNNER BAR -->
          <div id="compare-banner" class="hidden mb-3 p-3 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-indigo-500/10 rounded-xl border border-indigo-500/20 flex flex-col md:flex-row md:items-center justify-between gap-2.5">
            <div>
              <div class="text-xs font-bold text-indigo-600 dark:text-indigo-400 flex items-center gap-1.5">
                <span>⚡ Re-transcribe with Another Model Size:</span>
              </div>
              <p class="text-[11px] text-slate-500 dark:text-slate-400">Click any model below to immediately run and benchmark quality vs speed</p>
            </div>
            <div class="flex flex-wrap items-center gap-1.5" id="quick-model-buttons">
              <button onclick="runModelDirectly('tiny')" class="px-2.5 py-1 bg-white dark:bg-slate-800 hover:bg-indigo-50 hover:border-indigo-400 dark:hover:bg-slate-700 text-indigo-600 dark:text-indigo-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all">Tiny</button>
              <button onclick="runModelDirectly('base')" class="px-2.5 py-1 bg-white dark:bg-slate-800 hover:bg-indigo-50 hover:border-indigo-400 dark:hover:bg-slate-700 text-indigo-600 dark:text-indigo-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all">Base</button>
              <button onclick="runModelDirectly('small')" class="px-2.5 py-1 bg-white dark:bg-slate-800 hover:bg-indigo-50 hover:border-indigo-400 dark:hover:bg-slate-700 text-indigo-600 dark:text-indigo-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all">Small</button>
              <button onclick="runModelDirectly('medium')" class="px-2.5 py-1 bg-white dark:bg-slate-800 hover:bg-indigo-50 hover:border-indigo-400 dark:hover:bg-slate-700 text-indigo-600 dark:text-indigo-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all">Medium</button>
              <button onclick="runModelDirectly('large-v3')" class="px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold shadow-sm transition-all">Large-v3</button>
              <button onclick="openCompareWithSource(encodeURIComponent(currentSourceName))" class="px-2.5 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-bold shadow-sm transition-all">⚖️ Compare Runs</button>
            </div>
          </div>

          <div id="speaker-rename-bar" class="hidden mb-3 p-3 bg-slate-100 dark:bg-slate-950/80 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2">
            <div class="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-300">
              <span>🏷️ Rename Speakers (Real-time update across all exports)</span>
              <span id="speaker-count" class="text-[10px] text-indigo-500 font-mono">0 Speakers</span>
            </div>
            <div id="speaker-inputs" class="flex flex-wrap gap-2 pt-1"></div>
          </div>

          <div id="progress-box" class="hidden mb-4 p-3 bg-slate-100/80 dark:bg-slate-950/80 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2">
            <div class="flex items-center justify-between text-xs font-medium">
              <span id="progress-stage" class="text-indigo-600 dark:text-indigo-400 font-semibold flex items-center gap-1.5">
                <span id="spinner" class="inline-block w-3 h-3 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></span>
                <span id="progress-stage-text">Processing...</span>
              </span>
              <span id="progress-status" class="text-slate-500 dark:text-slate-400 font-mono text-[11px]">Elapsed: 0s</span>
            </div>
            <div class="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="progress-bar" class="bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 h-full w-0 transition-all duration-300"></div>
            </div>
          </div>

          <div id="virtual-viewport" class="flex-1 overflow-y-auto max-h-[480px] pr-2 text-sm relative">
            <div id="spacer-top" style="height: 0px;"></div>
            <div id="segments-container" class="space-y-3">
              <div class="text-center text-slate-400 dark:text-slate-600 text-xs py-24">Upload an audio file or paste a Google Drive / YouTube URL.</div>
            </div>
            <div id="spacer-bottom" style="height: 0px;"></div>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- HARD TOKEN GATE / LOCK SCREEN -->
  <div id="token-gate" class="fixed inset-0 z-[100] bg-slate-900/90 backdrop-blur-md flex items-center justify-center p-4">
    <div class="w-full max-w-sm bg-white dark:bg-slate-900 rounded-3xl p-8 border border-slate-200 dark:border-slate-800 shadow-2xl space-y-5 text-center">
      <div class="w-16 h-16 bg-indigo-600/10 text-indigo-600 dark:text-indigo-400 rounded-2xl flex items-center justify-center mx-auto text-3xl shadow-inner">
        🔒
      </div>
      <div>
        <h2 class="text-xl font-bold text-slate-800 dark:text-slate-100">Site Protected</h2>
        <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Please enter your authorized access token to unlock the AI Transcription dashboard.</p>
      </div>
      <div class="space-y-3 text-left">
        <div>
          <label class="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">Access Token</label>
          <input type="password" id="gate-token-input" placeholder="Enter Token" class="w-full bg-slate-50 dark:bg-slate-800/80 border border-slate-300 dark:border-slate-700 rounded-xl px-4 py-2.5 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all font-mono" />
        </div>
        <p id="gate-error-msg" class="text-xs text-rose-500 hidden font-medium text-center">Invalid access token. Please try again.</p>
        <button id="btn-unlock-gate" class="w-full bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] text-white font-semibold py-2.5 rounded-xl text-sm transition-all shadow-lg shadow-indigo-500/25 flex items-center justify-center gap-2">
          <span>Unlock Dashboard</span> <span>➜</span>
        </button>
      </div>
    </div>
  </div>

  <!-- TOKEN MODAL -->
  <div id="token-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="w-full max-w-sm bg-white dark:bg-slate-900 rounded-2xl p-6 border border-slate-200 dark:border-slate-800 shadow-2xl space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">🔑 Access Token</h3>
        <button id="btn-close-token" class="text-slate-400 hover:text-slate-600">✕</button>
      </div>
      <p class="text-xs text-slate-500 dark:text-slate-400">Enter your authorized API token to unlock transcription features.</p>
      <input type="password" id="token-input" placeholder="Enter API token" class="w-full bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500" />
      <button id="btn-save-token" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2 rounded-xl text-xs transition-colors">Save Token</button>
    </div>
  </div>

  <!-- HISTORY MODAL (Grouped by Source) -->
  <div id="history-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm hidden flex justify-end">
    <div class="w-full max-w-lg bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 h-full p-6 flex flex-col space-y-4 shadow-2xl">
      <div class="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-800">
        <div>
          <h3 class="font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">🕒 Transcription History</h3>
          <p class="text-[11px] text-slate-500 dark:text-slate-400">Saved runs grouped by audio source</p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="clearAllHistoryRecords()" class="text-xs text-rose-500 hover:text-rose-600 font-medium px-2 py-1 rounded hover:bg-rose-500/10 transition-colors">Clear All</button>
          <button id="btn-close-history" class="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-100 rounded-lg">✕</button>
        </div>
      </div>
      <div id="history-list" class="flex-1 overflow-y-auto space-y-3 pr-1 text-xs"></div>
    </div>
  </div>

  <!-- COMPARE MODAL -->
  <div id="compare-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="w-full max-w-5xl bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col max-h-[92vh] overflow-hidden">
      <!-- Modal Header -->
      <div class="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50/50 dark:bg-slate-950/50">
        <div>
          <h3 class="font-bold text-lg text-slate-800 dark:text-slate-100 flex items-center gap-2">
            <span>⚖️ Whisper Model Comparison & Benchmarks</span>
          </h3>
          <p class="text-xs text-slate-500 dark:text-slate-400">Compare transcription accuracy, processing speed, and word differences side-by-side</p>
        </div>
        <button id="btn-close-compare" class="text-slate-400 hover:text-slate-700 dark:hover:text-slate-100 text-lg p-1">✕</button>
      </div>

      <!-- Controls & Selectors -->
      <div class="px-6 py-3 border-b border-slate-200 dark:border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-4 items-center bg-slate-50/30 dark:bg-slate-950/30 text-xs">
        <div>
          <label class="block font-semibold text-slate-600 dark:text-slate-400 mb-1">Audio Source</label>
          <select id="compare-source-select" class="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 font-medium text-slate-800 dark:text-slate-200">
            <option value="">Select an audio source...</option>
          </select>
        </div>
        <div>
          <label class="block font-semibold text-indigo-600 dark:text-indigo-400 mb-1">Model A (Baseline)</label>
          <select id="compare-run-a" class="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 font-medium text-slate-800 dark:text-slate-200"></select>
        </div>
        <div>
          <label class="block font-semibold text-purple-600 dark:text-purple-400 mb-1">Model B (Comparison)</label>
          <select id="compare-run-b" class="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 font-medium text-slate-800 dark:text-slate-200"></select>
        </div>
      </div>

      <!-- Metrics Cards -->
      <div id="compare-metrics" class="px-6 py-4 border-b border-slate-200 dark:border-slate-800 grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="bg-indigo-50/60 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-900/50 rounded-xl p-3 text-center">
          <span class="text-[10px] font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider block">Text Similarity</span>
          <span id="metric-similarity" class="text-2xl font-black text-indigo-700 dark:text-indigo-300">--%</span>
          <span class="text-[10px] text-slate-500 block mt-0.5">Word sequence match</span>
        </div>
        <div class="bg-emerald-50/60 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/50 rounded-xl p-3 text-center">
          <span class="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider block">Speed & Processing</span>
          <div id="metric-speed" class="text-sm font-bold text-emerald-700 dark:text-emerald-300 mt-1">A: --s vs B: --s</div>
          <span id="metric-speedup" class="text-[10px] text-slate-500 block mt-0.5">--x faster</span>
        </div>
        <div class="bg-amber-50/60 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 rounded-xl p-3 text-center">
          <span class="text-[10px] font-semibold text-amber-600 dark:text-amber-400 uppercase tracking-wider block">Total Words</span>
          <div id="metric-words" class="text-sm font-bold text-amber-700 dark:text-amber-300 mt-1">A: 0 vs B: 0</div>
          <span id="metric-word-diff" class="text-[10px] text-slate-500 block mt-0.5">0 word diff</span>
        </div>
        <div class="bg-purple-50/60 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-900/50 rounded-xl p-3 text-center">
          <span class="text-[10px] font-semibold text-purple-600 dark:text-purple-400 uppercase tracking-wider block">Speakers</span>
          <div id="metric-speakers" class="text-sm font-bold text-purple-700 dark:text-purple-300 mt-1">A: 0 vs B: 0</div>
          <span class="text-[10px] text-slate-500 block mt-0.5">Detected turns</span>
        </div>
      </div>

      <!-- Diff Mode Switcher -->
      <div class="px-6 py-2 bg-slate-100/60 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs">
        <div class="flex items-center gap-3">
          <span class="font-medium text-slate-600 dark:text-slate-400">Diff Legend:</span>
          <span class="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400"><span class="w-2 h-2 rounded-full bg-emerald-500"></span> Added in B</span>
          <span class="inline-flex items-center gap-1 text-[11px] font-medium text-rose-600 dark:text-rose-400"><span class="w-2 h-2 rounded-full bg-rose-500"></span> Deleted from A</span>
          <span class="inline-flex items-center gap-1 text-[11px] font-medium text-amber-600 dark:text-amber-400"><span class="w-2 h-2 rounded-full bg-amber-500"></span> Substituted</span>
        </div>
        <div class="flex items-center gap-2">
          <button id="btn-toggle-view" onclick="toggleCompareView()" class="px-2.5 py-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md font-semibold text-slate-700 dark:text-slate-300 shadow-sm">
            Switch to Timeline View
          </button>
        </div>
      </div>

      <!-- Comparison Split Content -->
      <div class="flex-1 overflow-y-auto p-6 grid grid-cols-1 md:grid-cols-2 gap-6" id="compare-split-view">
        <!-- Panel A -->
        <div class="flex flex-col space-y-2 border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-slate-50/50 dark:bg-slate-950/50">
          <div class="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-2">
            <h4 id="label-model-a" class="font-bold text-sm text-indigo-600 dark:text-indigo-400">Model A</h4>
            <span id="badge-meta-a" class="text-[11px] text-slate-500 font-mono">--</span>
          </div>
          <div id="diff-text-a" class="flex-1 text-sm leading-relaxed whitespace-normal select-text space-y-1"></div>
        </div>

        <!-- Panel B -->
        <div class="flex flex-col space-y-2 border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-slate-50/50 dark:bg-slate-950/50">
          <div class="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-2">
            <h4 id="label-model-b" class="font-bold text-sm text-purple-600 dark:text-purple-400">Model B</h4>
            <span id="badge-meta-b" class="text-[11px] text-slate-500 font-mono">--</span>
          </div>
          <div id="diff-text-b" class="flex-1 text-sm leading-relaxed whitespace-normal select-text space-y-1"></div>
        </div>
      </div>

      <!-- Segment-by-segment Timeline View (Toggleable) -->
      <div class="flex-1 overflow-y-auto p-6 hidden space-y-3" id="compare-timeline-view">
        <div id="timeline-comparative-container" class="space-y-4 text-xs"></div>
      </div>
    </div>
  </div>

  <!-- MOM MODAL -->
  <div id="mom-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="w-full max-w-4xl bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col max-h-[92vh] overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-lg">📝</div>
          <div>
            <h3 class="font-bold text-base text-slate-800 dark:text-slate-100 flex items-center gap-2">
              <span>Minutes of Meeting (MOM)</span>
              <span class="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-700 dark:text-amber-300 font-bold border border-amber-500/30">FreeToken Qwen 3.8*</span>
            </h3>
            <p class="text-xs text-slate-500 dark:text-slate-400">Synthesized Executive Summary, Agenda Points, Decisions, & Action Items</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="copyMomMarkdown()" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow-sm flex items-center gap-1.5 transition-all">📋 Copy Markdown</button>
          <button onclick="downloadMomMarkdown()" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-semibold shadow-sm flex items-center gap-1.5 transition-all">💾 Download .mom.md</button>
          <button id="btn-close-mom" onclick="closeMomModal()" class="text-slate-400 hover:text-slate-700 dark:hover:text-slate-100 text-lg p-1">✕</button>
        </div>
      </div>
      <div class="p-6 flex-1 overflow-y-auto bg-slate-50/50 dark:bg-slate-950/50 space-y-3">
        <div id="mom-status" class="hidden p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl text-xs text-amber-700 dark:text-amber-300 font-medium flex items-center gap-2 animate-pulse">
          <span>🧠 Analyzing dialogue with FreeToken Qwen 3.8* engine...</span>
        </div>
        <div id="mom-content" class="prose dark:prose-invert max-w-none text-xs text-slate-800 dark:text-slate-200 whitespace-pre-wrap font-sans leading-relaxed bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm min-h-[300px]">
          Click "Generate MOM" to synthesize meeting minutes.
        </div>
      </div>
    </div>
  </div>

  <!-- REFINE MODAL -->
  <div id="refine-modal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="w-full max-w-4xl bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col max-h-[92vh] overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-gradient-to-r from-teal-500/10 via-emerald-500/10 to-teal-500/10">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-xl bg-teal-500/20 border border-teal-500/30 flex items-center justify-center text-lg">✨</div>
          <div>
            <h3 class="font-bold text-base text-slate-800 dark:text-slate-100 flex items-center gap-2">
              <span>AI Transcript Refiner & Polisher</span>
              <span class="text-[10px] px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-700 dark:text-teal-300 font-bold border border-teal-500/30">FreeToken Qwen 3.8*</span>
            </h3>
            <p class="text-xs text-slate-500 dark:text-slate-400">Polished grammar, punctuation, sentence boundaries, and disfluency cleanup</p>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="copyRefinedText()" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow-sm flex items-center gap-1.5 transition-all">📋 Copy Refined</button>
          <button onclick="downloadRefinedText()" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg text-xs font-semibold shadow-sm flex items-center gap-1.5 transition-all">💾 Download .refined.txt</button>
          <button id="btn-close-refine" onclick="closeRefineModal()" class="text-slate-400 hover:text-slate-700 dark:hover:text-slate-100 text-lg p-1">✕</button>
        </div>
      </div>
      <div class="p-6 flex-1 overflow-y-auto bg-slate-50/50 dark:bg-slate-950/50 space-y-3">
        <div id="refine-status" class="hidden p-3 bg-teal-500/10 border border-teal-500/20 rounded-xl text-xs text-teal-700 dark:text-teal-300 font-medium flex items-center gap-2 animate-pulse">
          <span>🧠 Polishing transcript with FreeToken Qwen 3.8* engine...</span>
        </div>
        <div id="refine-content" class="text-xs text-slate-800 dark:text-slate-200 whitespace-pre-wrap font-sans leading-relaxed bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm min-h-[300px]">
          Click "Refine" to generate polished transcript.
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentResult = { segments: [] }, currentJobId = null, currentSourceName = 'transcript', activeTab = 'file', timerInterval = null, startTime = 0;
    let cachedFile = null;
    let appToken = localStorage.appToken || '';
    let compareSourcesData = [];
    let compareViewMode = 'diff';
    const ROW_HEIGHT = 80, BUFFER = 25;
    const COLORS = [
      { bg: 'bg-indigo-500/15', text: 'text-indigo-600 dark:text-indigo-400', border: 'border-indigo-500/30' },
      { bg: 'bg-emerald-500/15', text: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/30' },
      { bg: 'bg-amber-500/15', text: 'text-amber-600 dark:text-amber-400', border: 'border-amber-500/30' },
      { bg: 'bg-rose-500/15', text: 'text-rose-600 dark:text-rose-400', border: 'border-rose-500/30' },
      { bg: 'bg-cyan-500/15', text: 'text-cyan-600 dark:text-cyan-400', border: 'border-cyan-500/30' }
    ];

    const tabFile = document.getElementById('tab-file'), tabUrl = document.getElementById('tab-url');
    const panelFile = document.getElementById('panel-file'), panelUrl = document.getElementById('panel-url');
    const fileInput = document.getElementById('file-input'), dropzone = document.getElementById('dropzone');
    const urlInput = document.getElementById('url-input'), fileInfo = document.getElementById('file-info');
    const btnTranscribe = document.getElementById('btn-transcribe'), segmentsContainer = document.getElementById('segments-container');
    const viewport = document.getElementById('virtual-viewport'), spacerTop = document.getElementById('spacer-top'), spacerBottom = document.getElementById('spacer-bottom');
    const metaLabel = document.getElementById('meta-label'), exportActions = document.getElementById('export-actions'), streamBadge = document.getElementById('stream-badge'), btnResume = document.getElementById('btn-resume-action');
    const historyModal = document.getElementById('history-modal'), historyList = document.getElementById('history-list'), exportStyle = document.getElementById('export-style');
    const compareBanner = document.getElementById('compare-banner'), modelSelect = document.getElementById('model-select');
    const familySelect = document.getElementById('family-select'), variantSelect = document.getElementById('variant-select'), computeTypeSelect = document.getElementById('compute-type-select');
    const adaptiveKnobs = document.getElementById('adaptive-knobs'), quickModelButtons = document.getElementById('quick-model-buttons');
    const badgeCache = document.getElementById('badge-cache'), badgeParams = document.getElementById('badge-params'), badgeVram = document.getElementById('badge-vram'), badgeSpeed = document.getElementById('badge-speed');
    const compareModal = document.getElementById('compare-modal'), compareSourceSelect = document.getElementById('compare-source-select'), compareRunA = document.getElementById('compare-run-a'), compareRunB = document.getElementById('compare-run-b');
    const progressBox = document.getElementById('progress-box'), progressStageText = document.getElementById('progress-stage-text'), progressStatus = document.getElementById('progress-status'), progressBar = document.getElementById('progress-bar'), spinner = document.getElementById('spinner');
    const renameBar = document.getElementById('speaker-rename-bar'), speakerInputs = document.getElementById('speaker-inputs'), speakerCount = document.getElementById('speaker-count');
    const tokenModal = document.getElementById('token-modal'), tokenInput = document.getElementById('token-input'), tokenLabel = document.getElementById('token-label');
    const tokenGate = document.getElementById('token-gate'), gateTokenInput = document.getElementById('gate-token-input'), gateErrorMsg = document.getElementById('gate-error-msg'), btnUnlockGate = document.getElementById('btn-unlock-gate');
    const toast = document.getElementById('toast'), toastMsg = document.getElementById('toast-msg');
    let allModelsCatalog = [];

    function showToast(msg) { toastMsg.textContent = msg; toast.classList.remove('translate-y-[-100px]', 'opacity-0'); setTimeout(() => toast.classList.add('translate-y-[-100px]', 'opacity-0'), 2500); }

    document.getElementById('btn-theme-toggle').onclick = () => { const isDark = document.documentElement.classList.toggle('dark'); localStorage.theme = isDark ? 'dark' : 'light'; };
    document.getElementById('btn-open-history').onclick = () => { historyModal.classList.remove('hidden'); loadHistory(); };
    document.getElementById('btn-close-history').onclick = () => historyModal.classList.add('hidden');
    historyModal.onclick = (e) => { if (e.target === historyModal) historyModal.classList.add('hidden'); };

    document.getElementById('btn-open-compare').onclick = () => openCompareModal();
    document.getElementById('btn-close-compare').onclick = () => compareModal.classList.add('hidden');
    compareModal.onclick = (e) => { if (e.target === compareModal) compareModal.classList.add('hidden'); };

    function updateTokenUI() {
      tokenLabel.textContent = appToken ? '🔑 Token: ••••••••' : '🔑 Set Token';
    }

    async function verifyAndApplyToken(candidateToken) {
      if (!candidateToken) return false;
      try {
        const res = await fetch(`/api/auth/verify?token=${encodeURIComponent(candidateToken)}`);
        if (res.ok) {
          appToken = candidateToken;
          localStorage.appToken = appToken;
          updateTokenUI();
          tokenGate.classList.add('hidden');
          gateErrorMsg.classList.add('hidden');
          loadHistory(true);
          loadModelCatalog();
          return true;
        }
      } catch (e) {
        console.error('Token verification error:', e);
      }
      return false;
    }

    async function handleUnlockGate() {
      const entered = gateTokenInput.value.trim();
      if (!entered) {
        gateErrorMsg.textContent = 'Please enter an access token.';
        gateErrorMsg.classList.remove('hidden');
        return;
      }
      btnUnlockGate.disabled = true;
      btnUnlockGate.innerHTML = '<span>Verifying...</span>';
      const success = await verifyAndApplyToken(entered);
      btnUnlockGate.disabled = false;
      btnUnlockGate.innerHTML = '<span>Unlock Dashboard</span> <span>➜</span>';
      if (!success) {
        gateErrorMsg.textContent = 'Invalid access token. Please try again.';
        gateErrorMsg.classList.remove('hidden');
        gateTokenInput.classList.add('border-rose-500');
        setTimeout(() => gateTokenInput.classList.remove('border-rose-500'), 2000);
      } else {
        showToast('Access Granted');
      }
    }

    btnUnlockGate.onclick = handleUnlockGate;
    gateTokenInput.onkeydown = (e) => { if (e.key === 'Enter') handleUnlockGate(); };

    document.getElementById('btn-token-modal').onclick = () => { tokenInput.value = appToken; tokenModal.classList.remove('hidden'); };
    document.getElementById('btn-close-token').onclick = () => tokenModal.classList.add('hidden');
    document.getElementById('btn-save-token').onclick = async () => {
      const candidate = tokenInput.value.trim();
      const ok = await verifyAndApplyToken(candidate);
      if (ok) {
        tokenModal.classList.add('hidden');
        showToast('Token updated');
      } else {
        showToast('Invalid token');
      }
    };

    // Initial auth gate verification
    (async function initAuth() {
      const stored = localStorage.appToken;
      if (stored) {
        const valid = await verifyAndApplyToken(stored);
        if (!valid) {
          localStorage.removeItem('appToken');
          appToken = '';
          tokenGate.classList.remove('hidden');
        }
      } else {
        tokenGate.classList.remove('hidden');
      }
    })();

    async function loadModelCatalog() {
      try {
        const res = await fetch('/api/models');
        if (res.ok) {
          const data = await res.json();
          allModelsCatalog = [...(data.local || []), ...(data.cloud || [])];
          renderFamilyOptions();
        }
      } catch (e) {
        console.error('Failed to load model catalog:', e);
      }
    }

    function renderFamilyOptions() {
      const families = [...new Set(allModelsCatalog.map(m => m.family))];
      familySelect.innerHTML = '';
      families.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f;
        opt.textContent = f;
        if (f.includes('Faster-Whisper')) opt.selected = true;
        familySelect.appendChild(opt);
      });
      onFamilySelectChange();
    }

    function onFamilySelectChange() {
      const selectedFam = familySelect.value;
      const variants = allModelsCatalog.filter(m => m.family === selectedFam);
      variantSelect.innerHTML = '';
      variants.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.display_name || m.name} (${m.params})`;
        if (m.name === 'base' || m.name === 'base.en' || variants.length === 1) opt.selected = true;
        variantSelect.appendChild(opt);
      });
      onVariantSelectChange();
      updateQuickButtons(selectedFam);
    }

    function onVariantSelectChange() {
      const modelName = variantSelect.value;
      const modelObj = allModelsCatalog.find(m => m.name === modelName) || { name: modelName, params: '74M', vram: '~1 GB', speed_factor: '~16x', is_cached: true, capabilities: ['local', 'gpu'] };
      
      modelSelect.value = modelName;
      btnTranscribe.textContent = `Start Transcription (${modelName.toUpperCase()})`;

      const quantOptions = modelObj.quantization_options || ['float16', 'int8', 'int8_float16'];
      computeTypeSelect.innerHTML = '<option value="default" selected>Auto / Default</option>';
      quantOptions.forEach(q => {
        const opt = document.createElement('option');
        opt.value = q;
        opt.textContent = q.toUpperCase();
        computeTypeSelect.appendChild(opt);
      });

      badgeParams.textContent = `${modelObj.params || '--'} params`;
      badgeVram.textContent = `${modelObj.vram || '--'} VRAM`;
      badgeSpeed.textContent = `${modelObj.speed_factor || '--'} RTF`;
      if (modelObj.is_cached) {
        badgeCache.textContent = '● Cached';
        badgeCache.className = 'px-1.5 py-0.5 rounded font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20';
      } else {
        badgeCache.textContent = '📥 Auto-Download';
        badgeCache.className = 'px-1.5 py-0.5 rounded font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20';
      }

      renderAdaptiveKnobs(modelObj);
    }

    function renderAdaptiveKnobs(modelObj) {
      const fam = modelObj.family || '';
      adaptiveKnobs.innerHTML = '';

      if (fam.includes('Whisper')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-medium text-slate-700 dark:text-slate-300">Beam Search Size</span>
            <select id="knob-beam-size" class="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-2 py-0.5 text-xs text-slate-800 dark:text-slate-200">
              <option value="1">1 (Greedy / Fastest)</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="5" selected>5 (Standard)</option>
              <option value="8">8 (High Accuracy)</option>
            </select>
          </div>
          <div class="flex items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800/80">
            <span class="font-medium text-slate-700 dark:text-slate-300">Silero VAD Filter</span>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="knob-vad-filter" checked class="sr-only peer">
              <div class="w-7 h-4 bg-slate-300 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>`;
      } else if (fam.includes('SenseVoice')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-medium text-slate-700 dark:text-slate-300">Inverse Text Norm (ITN)</span>
            <label class="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" id="knob-use-itn" checked class="sr-only peer">
              <div class="w-7 h-4 bg-slate-300 peer-focus:outline-none rounded-full peer dark:bg-slate-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>
          <div class="flex items-center gap-1.5 pt-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">
            <span>🎭 Emotion (SER) & Audio Events (AED) Enabled</span>
          </div>`;
      } else if (fam.includes('MMS')) {
        adaptiveKnobs.innerHTML = `
          <div>
            <label class="block font-medium text-slate-700 dark:text-slate-300 mb-1">MMS Target Adapter Code</label>
            <input type="text" id="knob-mms-lang" placeholder="e.g. ind, eng, fra, spa, jav" class="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-200">
          </div>`;
      } else if (fam.includes('OmniASR')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between text-indigo-600 dark:text-indigo-400 font-medium">
            <span>🌐 1,600+ Languages Acoustic CTC</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">Omnilingual ASR</span>
          </div>`;
      } else if (fam.includes('CTC') || fam.includes('Wav2Vec2')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-medium text-slate-700 dark:text-slate-300">Chunk Window Size</span>
            <select id="knob-chunk-size" class="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-2 py-0.5 text-xs text-slate-800 dark:text-slate-200">
              <option value="15.0">15s (Low Latency)</option>
              <option value="30.0" selected>30s (Default)</option>
              <option value="45.0">45s</option>
              <option value="60.0">60s</option>
            </select>
          </div>`;
      } else if (fam.includes('Moonshine')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between text-indigo-600 dark:text-indigo-400 font-medium">
            <span>⚡ Zero-Overhead ONNX Runtime</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20">Edge Optimized</span>
          </div>`;
      } else if (fam.includes('FireRed')) {
        const isLlm = (modelObj.name || '').includes('llm') || (modelObj.name || '').includes('9b');
        if (isLlm) {
          adaptiveKnobs.innerHTML = `
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-slate-700 dark:text-slate-300">Temperature</span>
              <input type="number" id="knob-temperature" min="0.0" max="1.0" step="0.1" value="0.0" class="w-16 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-1.5 py-0.5 text-xs text-slate-800 dark:text-slate-200">
            </div>
            <div class="flex items-center gap-1.5 pt-1 text-[11px] text-amber-600 dark:text-amber-400 font-medium">
              <span>🔥 Qwen2-7B LLM Speech Interaction</span>
            </div>`;
        } else {
          adaptiveKnobs.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="font-medium text-slate-700 dark:text-slate-300">Beam Search Size</span>
              <select id="knob-beam-size" class="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-2 py-0.5 text-xs text-slate-800 dark:text-slate-200">
                <option value="1">1 (Greedy / Fast)</option>
                <option value="3">3</option>
                <option value="5" selected>5 (Standard)</option>
                <option value="10">10 (Accurate)</option>
              </select>
            </div>
            <div class="flex items-center gap-1.5 pt-1 text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
              <span>🔥 Industrial Conformer-AED Speech Recognition</span>
            </div>`;
        }
      } else if (fam.includes('VoiceMem')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between text-xs">
            <span class="font-medium text-slate-700 dark:text-slate-300">Dual-Brain Cognition</span>
            <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/20">Active</span>
          </div>
          <div class="flex items-center gap-1.5 pt-1 text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
            <span>🧠 Emotion (SER) + Voiceprint + Factual Memory</span>
          </div>`;
      } else if (fam.includes('Whisper.cpp')) {
        adaptiveKnobs.innerHTML = `
          <div class="flex items-center justify-between text-xs">
            <span class="font-medium text-slate-700 dark:text-slate-300">CPU SIMD Threads</span>
            <input type="number" id="knob-threads" min="1" max="32" value="4" class="w-14 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-1.5 py-0.5 text-xs text-slate-800 dark:text-slate-200">
          </div>
          <div class="flex items-center gap-1.5 pt-1 text-[11px] text-amber-600 dark:text-amber-400 font-medium">
            <span>⚡ High-Efficiency GGML / GGUF C++ Engine</span>
          </div>`;
      } else if (fam.includes('NVIDIA') || fam.includes('NeMo')) {
        const isAudex = (modelObj.name || '').includes('audex');
        if (isAudex) {
          adaptiveKnobs.innerHTML = `
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-slate-700 dark:text-slate-300">Execution Mode</span>
              <select id="knob-audex-mode" class="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-2 py-0.5 text-xs text-slate-800 dark:text-slate-200">
                <option value="instruct" selected>⚡ Instruct Mode (Verbatim ASR)</option>
                <option value="thinking">🤔 Thinking Mode (Reasoning)</option>
              </select>
            </div>
            <div class="flex items-center gap-1.5 pt-1 text-[11px] text-teal-600 dark:text-teal-400 font-medium">
              <span>🧠 2B Compact Unified Audio-Text LLM</span>
            </div>`;
        } else {
          adaptiveKnobs.innerHTML = `
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-slate-700 dark:text-slate-300">Fast-Conformer TDT</span>
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/20">ONNX Ready</span>
            </div>
            <div class="flex items-center gap-1.5 pt-1 text-[11px] text-teal-600 dark:text-teal-400 font-medium">
              <span>🦅 NVIDIA Parakeet / Nemotron Low-Latency ASR</span>
            </div>`;
        }
      } else {
        adaptiveKnobs.innerHTML = `<span class="text-slate-400">Standard inference settings active.</span>`;
      }
    }

    function updateQuickButtons(activeFam) {
      const familyModels = allModelsCatalog.filter(m => m.family === activeFam && m.is_local);
      const topOtherModels = allModelsCatalog.filter(m => m.family !== activeFam && m.is_local && ['sensevoice-small', 'turbo', 'moonshine-base', 'indonesian-wav2vec2-regional', 'meta-omnilingual-asr'].includes(m.name)).slice(0, 2);
      
      quickModelButtons.innerHTML = '';
      familyModels.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'px-2.5 py-1 bg-white dark:bg-slate-800 hover:bg-indigo-50 hover:border-indigo-400 dark:hover:bg-slate-700 text-indigo-600 dark:text-indigo-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all';
        btn.textContent = m.display_name ? m.display_name.replace('Whisper ', '').replace(' (ID)', '') : m.name;
        btn.onclick = () => selectAndRunModel(m.name, m.family);
        quickModelButtons.appendChild(btn);
      });

      topOtherModels.forEach(m => {
        const btn = document.createElement('button');
        btn.className = 'px-2.5 py-1 bg-slate-100 dark:bg-slate-900 hover:bg-purple-50 hover:border-purple-400 dark:hover:bg-purple-950/40 text-purple-600 dark:text-purple-400 border border-slate-300 dark:border-slate-700 rounded-lg text-xs font-bold shadow-sm transition-all';
        btn.textContent = `⚡ ${m.display_name || m.name}`;
        btn.onclick = () => selectAndRunModel(m.name, m.family);
        quickModelButtons.appendChild(btn);
      });

      const cmpBtn = document.createElement('button');
      cmpBtn.className = 'px-2.5 py-1 bg-purple-600 hover:bg-purple-500 text-white rounded-lg text-xs font-bold shadow-sm transition-all';
      cmpBtn.textContent = '⚖️ Compare Runs';
      cmpBtn.onclick = () => openCompareWithSource(encodeURIComponent(currentSourceName));
      quickModelButtons.appendChild(cmpBtn);
    }

    function selectAndRunModel(modelName, modelFamily) {
      if (modelFamily && familySelect.value !== modelFamily) {
        familySelect.value = modelFamily;
        onFamilySelectChange();
      }
      variantSelect.value = modelName;
      onVariantSelectChange();
      executeStream(false, modelName);
    }

    familySelect.onchange = onFamilySelectChange;
    variantSelect.onchange = onVariantSelectChange;
    loadModelCatalog();

    function authHeaders(h = {}) { return appToken ? { ...h, 'X-API-Token': appToken } : h; }

    tabFile.onclick = () => { activeTab = 'file'; tabFile.className = 'text-xs font-semibold pb-1 border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400'; tabUrl.className = 'text-xs font-semibold pb-1 border-b-2 border-transparent text-slate-500 dark:text-slate-400'; panelFile.classList.remove('hidden'); panelUrl.classList.add('hidden'); checkTranscribeButton(); };
    tabUrl.onclick = () => { activeTab = 'url'; tabUrl.className = 'text-xs font-semibold pb-1 border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400'; tabFile.className = 'text-xs font-semibold pb-1 border-b-2 border-transparent text-slate-500 dark:text-slate-400'; panelUrl.classList.remove('hidden'); panelFile.classList.add('hidden'); checkTranscribeButton(); };

    function checkTranscribeButton() {
      if (activeTab === 'file') {
        btnTranscribe.disabled = !(fileInput.files.length || cachedFile || (currentSourceName && currentSourceName !== 'transcript'));
      } else {
        btnTranscribe.disabled = !urlInput.value.trim();
      }
    }

    dropzone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
      const f = e.target.files[0];
      if (f) {
        cachedFile = f;
        currentSourceName = f.name;
        fileInfo.textContent = `${f.name} (${(f.size/1048576).toFixed(2)} MB)`;
        fileInfo.classList.remove('hidden');
        btnTranscribe.disabled = false;
      }
    };
    urlInput.oninput = () => { checkTranscribeButton(); };
    modelSelect.onchange = () => {
      checkTranscribeButton();
      const m = modelSelect.value.toUpperCase();
      btnTranscribe.textContent = `Start Transcription (${m})`;
    };

    function resumeTranscription() { btnResume.classList.add('hidden'); executeStream(true); }

    function clearCurrentTimeline() {
      currentResult = { segments: [] }; currentJobId = null; cachedFile = null; fileInput.value = ''; fileInfo.classList.add('hidden');
      segmentsContainer.innerHTML = '<div class="text-center text-slate-400 dark:text-slate-600 text-xs py-24">Upload an audio file or paste a Google Drive / YouTube URL.</div>';
      metaLabel.textContent = 'Ready to transcribe.'; exportActions.classList.add('opacity-30', 'pointer-events-none');
      progressBox.classList.add('hidden'); renameBar.classList.add('hidden'); btnResume.classList.add('hidden');
      compareBanner.classList.add('hidden'); btnTranscribe.disabled = true; btnTranscribe.textContent = 'Start Transcription';
      spacerTop.style.height = '0px'; spacerBottom.style.height = '0px'; showToast('Timeline cleared');
    }

    function runModelDirectly(modelName) {
      modelSelect.value = modelName;
      executeStream(false, modelName);
    }

    function deleteSegment(idx) {
      if (idx >= 0 && idx < currentResult.segments.length) {
        currentResult.segments.splice(idx, 1);
        updateVirtualWindow(); renderSpeakerRenameBar();
        if (currentJobId) fetch(`/api/history/${currentJobId}`, { method: 'PATCH', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(currentResult) }).catch(()=>{});
        showToast('Segment deleted');
      }
    }

    function getSpeakerStyle(speaker) {
      let idx = 0; const numMatch = String(speaker).match(/\\d+/);
      if (numMatch) idx = (parseInt(numMatch[0], 10) - 1) % COLORS.length;
      else idx = Math.abs(String(speaker).split('').reduce((a,c)=>a+c.charCodeAt(0),0)) % COLORS.length;
      return COLORS[Math.max(0, idx)];
    }

    function getSpeakerInitials(speaker) {
      if (!speaker) return 'S1';
      const clean = String(speaker).trim();
      const spkMatch = clean.match(/^SPEAKER_?(\d+)$/i) || clean.match(/^Speaker\s*(\d+)$/i);
      if (spkMatch) return 'S' + parseInt(spkMatch[1], 10);
      const words = clean.split(/[\s._-]+/).filter(w => w.length > 0);
      if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
      if (words.length === 1) {
        const w = words[0];
        return w.length >= 2 ? w.substring(0, 2).toUpperCase() : (w[0] + '1').toUpperCase();
      }
      return clean.substring(0, 2).toUpperCase() || 'SP';
    }

    let renameDebounceTimer = null;
    function renderSpeakerRenameBar() {
      const distinct = [...new Set(currentResult.segments.map(s => s.speaker || 'Speaker 1'))];
      currentResult.speakers = distinct;
      if (!distinct.length) { renameBar.classList.add('hidden'); return; }
      renameBar.classList.remove('hidden');
      speakerCount.textContent = `${distinct.length} Speaker${distinct.length > 1 ? 's' : ''}`;
      speakerInputs.innerHTML = '';
      distinct.forEach(spk => {
        let currentAssignedName = spk;
        const style = getSpeakerStyle(spk);
        const wrapper = document.createElement('div');
        wrapper.className = `flex items-center gap-1.5 px-2 py-1 rounded-lg border ${style.bg} ${style.border}`;
        wrapper.innerHTML = `<span class="avatar-badge inline-flex items-center justify-center min-w-[24px] h-5 px-1 rounded text-[10px] font-extrabold ${style.bg} ${style.text} border ${style.border}" title="${spk}">${getSpeakerInitials(spk)}</span><span class="text-slate-400 text-xs">➜</span><input type="text" value="${spk}" placeholder="Speaker name..." class="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded px-1.5 py-0.5 text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-indigo-500 w-28 font-medium" />`;
        
        const inputEl = wrapper.querySelector('input');
        const badgeEl = wrapper.querySelector('.avatar-badge');
        inputEl.oninput = (e) => {
          const newName = e.target.value.trim();
          if (!newName || newName === currentAssignedName) return;
          badgeEl.textContent = getSpeakerInitials(newName);
          badgeEl.title = newName;
          currentResult.segments.forEach(s => {
            if (s.speaker === currentAssignedName) s.speaker = newName;
          });
          currentAssignedName = newName;
          currentResult.speakers = [...new Set(currentResult.segments.map(s => s.speaker || 'Speaker 1'))];
          updateVirtualWindow();
          if (currentJobId) {
            if (renameDebounceTimer) clearTimeout(renameDebounceTimer);
            renameDebounceTimer = setTimeout(() => {
              fetch(`/api/history/${currentJobId}`, {
                method: 'PATCH',
                headers: authHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify(currentResult),
              }).catch(()=>{});
            }, 300);
          }
        };
        speakerInputs.appendChild(wrapper);
      });
    }

    viewport.onscroll = () => updateVirtualWindow();

    function updateVirtualWindow() {
      const segs = currentResult.segments;
      if (!segs.length) { spacerTop.style.height = '0px'; spacerBottom.style.height = '0px'; return; }
      const scrollTop = viewport.scrollTop;
      const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER);
      const endIdx = Math.min(segs.length, startIdx + BUFFER * 2);

      spacerTop.style.height = `${startIdx * ROW_HEIGHT}px`;
      spacerBottom.style.height = `${(segs.length - endIdx) * ROW_HEIGHT}px`;

      segmentsContainer.innerHTML = '';
      for (let i = startIdx; i < endIdx; i++) {
        const seg = segs[i];
        const spkName = seg.speaker || 'Speaker 1';
        const spkInitials = getSpeakerInitials(spkName);
        const isLatest = (i === segs.length - 1);
        const div = document.createElement('div');
        const style = getSpeakerStyle(spkName);
        div.className = `group relative p-3 rounded-xl border transition-all ${isLatest ? 'bg-indigo-50/70 dark:bg-indigo-950/40 border-indigo-500/60 shadow-sm shadow-indigo-500/10 ring-1 ring-indigo-500/30' : 'bg-slate-50 dark:bg-slate-950/60 border-slate-200 dark:border-slate-800/80 hover:border-indigo-400 dark:hover:border-slate-700'}`;
        const latestBadge = isLatest ? `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 animate-pulse">● Latest</span>` : '';
        const emotionBadge = (seg.emotion && seg.emotion !== 'NEUTRAL') ? `<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">🎭 ${seg.emotion}</span>` : '';
        const eventBadges = (seg.events && seg.events.length) ? seg.events.map(e => `<span class="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">🔔 ${e}</span>`).join('') : '';
        div.innerHTML = `<div class="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1"><div class="flex items-center gap-2"><span class="inline-flex items-center justify-center min-w-[26px] h-6 px-1.5 rounded-md font-bold text-[11px] ${style.bg} ${style.text} border ${style.border} shadow-xs cursor-default tracking-wide" title="${spkName}">${spkInitials}</span>${emotionBadge}${eventBadges}${latestBadge}</div><div class="flex items-center gap-2"><span class="font-mono text-slate-400 dark:text-slate-500">${formatTime(seg.start)} → ${formatTime(seg.end)}</span><button onclick="deleteSegment(${i})" class="opacity-0 group-hover:opacity-100 text-rose-400 hover:text-rose-600 p-0.5 rounded transition-opacity" title="Delete segment">✕</button></div></div><div class="text-slate-800 dark:text-slate-200 leading-relaxed">${seg.text}</div>`;
        segmentsContainer.appendChild(div);
      }
    }

    btnTranscribe.onclick = () => executeStream(false);

    async function executeStream(isResume = false, forcedModel = null) {
      if (!appToken) { tokenModal.classList.remove('hidden'); showToast('Please enter access token'); return; }
      const chosenModel = forcedModel || (variantSelect && variantSelect.value ? variantSelect.value : (modelSelect ? modelSelect.value : 'base'));
      if (modelSelect) modelSelect.value = chosenModel;
      const isForce = document.getElementById('chk-force') && document.getElementById('chk-force').checked;
      const formData = new FormData();
      formData.append('model', chosenModel);
      formData.append('language', document.getElementById('lang-input').value.trim() || '');
      formData.append('diarize', 'true');
      formData.append('token', appToken);
      if (isForce) formData.append('force', 'true');

      if (computeTypeSelect) formData.append('compute_type', computeTypeSelect.value);
      const beamInput = document.getElementById('knob-beam-size');
      if (beamInput) formData.append('beam_size', beamInput.value);
      const vadInput = document.getElementById('knob-vad-filter');
      if (vadInput) formData.append('vad_filter', vadInput.checked ? 'true' : 'false');
      const itnInput = document.getElementById('knob-use-itn');
      if (itnInput) formData.append('use_itn', itnInput.checked ? 'true' : 'false');
      const chunkInput = document.getElementById('knob-chunk-size');
      if (chunkInput) formData.append('chunk_length_s', chunkInput.value);
      const mmsLangInput = document.getElementById('knob-mms-lang');
      if (mmsLangInput && mmsLangInput.value.trim()) formData.append('target_lang', mmsLangInput.value.trim());
      const tempInput = document.getElementById('knob-temperature');
      if (tempInput) formData.append('temperature', tempInput.value);

      if (isResume && currentJobId) {
        formData.append('resume_job_id', currentJobId);
      } else {
        if (activeTab === 'file') {
          if (fileInput.files && fileInput.files[0]) {
            cachedFile = fileInput.files[0];
            formData.append('file', cachedFile);
            currentSourceName = cachedFile.name;
          } else if (cachedFile) {
            formData.append('file', cachedFile);
            currentSourceName = cachedFile.name;
          } else if (currentSourceName && currentSourceName !== 'transcript') {
            formData.append('source_name', currentSourceName);
          }
        } else {
          const u = urlInput.value.trim() || currentSourceName;
          formData.append('url', u);
          currentSourceName = u;
        }
        currentResult = { segments: [] }; currentJobId = null; segmentsContainer.innerHTML = '';
      }

      btnTranscribe.disabled = true; streamBadge.classList.remove('hidden'); btnResume.classList.add('hidden');
      compareBanner.classList.add('hidden');
      metaLabel.textContent = isResume ? 'Resuming transcription...' : `[${chosenModel.toUpperCase()}] Preparing audio...`;
      renameBar.classList.add('hidden'); progressBox.classList.remove('hidden'); progressBar.style.width = '15%';
      spinner.classList.remove('hidden');
      progressStageText.textContent = isResume ? '▶️ Resuming from last checkpoint...' : (activeTab === 'url' ? '⬇️ Downloading Audio...' : '⚙️ Normalizing Audio...');
      startTime = Date.now();
      if (timerInterval) clearInterval(timerInterval);
      timerInterval = setInterval(() => { const elap = Math.floor((Date.now() - startTime)/1000); progressStatus.textContent = `Elapsed: ${elap}s`; }, 1000);

      try {
        const response = await fetch(`/api/transcribe-stream?token=${encodeURIComponent(appToken)}`, { method: 'POST', headers: authHeaders(), body: formData });
        if (response.status === 401) { tokenGate.classList.remove('hidden'); localStorage.removeItem('appToken'); appToken = ''; throw new Error('Unauthorized: Invalid Token'); }
        if (!response.ok) { const errText = await response.text(); throw new Error(errText || `Server error ${response.status}`); }
        const reader = response.body.getReader(), decoder = new TextDecoder('utf-8');
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n'); buffer = lines.pop();
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const msg = JSON.parse(line.slice(6));
              if (msg.type === 'progress') {
                const p = msg.data;
                if (p.stage === 'downloading') {
                  const pct = p.percent ? p.percent.toFixed(1) : '25';
                  progressBar.style.width = `${pct}%`;
                  if (p.downloaded) {
                    const dlMb = (p.downloaded / 1048576).toFixed(1), totMb = p.total ? (p.total / 1048576).toFixed(1) + ' MB' : 'Unknown';
                    progressStageText.textContent = `⬇️ Downloading Audio (${dlMb} MB / ${totMb})`;
                  } else {
                    progressStageText.textContent = p.message || '📥 Downloading Model to Local Server Cache...';
                  }
                } else if (p.stage === 'audio_prep') {
                  progressStageText.textContent = p.message || '⚙️ Normalizing Audio (16kHz WAV)...';
                  progressBar.style.width = '30%';
                } else if (p.stage === 'vad_scan') {
                  progressStageText.textContent = `🔍 Scanning Speech & VAD (Total: ${formatNaturalDuration(p.duration)})...`;
                  progressBar.style.width = '50%';
                } else if (p.stage === 'transcribing') {
                  const pct = p.percent.toFixed(1);
                  progressBar.style.width = `${pct}%`;
                  progressStageText.textContent = `🎙️ [${chosenModel}] Transcribing Speech (Total: ${formatNaturalDuration(p.duration)})`;
                  const elap = Math.max(1, Math.floor((Date.now() - startTime)/1000));
                  const curAudio = p.current_time || 0, totAudio = p.duration || 0;
                  const speed = (curAudio > 0 && elap > 0) ? (curAudio / elap) : 0;
                  const remAudio = Math.max(0, totAudio - curAudio);
                  const etaSec = speed > 0 ? Math.floor(remAudio / speed) : 0;
                  const etaStr = etaSec > 0 ? ` • ETA: ~${formatNaturalDuration(etaSec)}` : '';
                  const speedStr = speed > 0 ? ` (${speed.toFixed(1)}x)` : '';
                  progressStatus.textContent = `${formatNaturalDuration(curAudio)} / ${formatNaturalDuration(totAudio)} (${pct}%) • Elapsed: ${elap}s${etaStr}${speedStr}`;
                  if (p.segment && !currentResult.segments.some(s => s.start === p.segment.start && s.end === p.segment.end)) {
                    currentResult.segments.push(p.segment);
                    updateVirtualWindow();
                    viewport.scrollTop = viewport.scrollHeight;
                  }
                }
              } else if (msg.type === 'done') {
                if (timerInterval) clearInterval(timerInterval);
                currentResult = msg.data; currentJobId = msg.job_id || null;
                const elap = msg.processing_time || Math.max(1, Math.floor((Date.now() - startTime)/1000));
                const totAudio = msg.data.duration || 0;
                const speed = totAudio > 0 ? (totAudio / elap).toFixed(1) : '1.0';
                progressBar.style.width = '100%'; spinner.classList.add('hidden');
                progressStageText.textContent = `✔ [${chosenModel}] Completed!`;
                progressStatus.textContent = `${formatNaturalDuration(totAudio)} audio transcribed in ${elap}s (${speed}x real-time)`;
                metaLabel.textContent = `Model: ${chosenModel.toUpperCase()} | Lang: ${msg.data.language.toUpperCase()} | Duration: ${formatNaturalDuration(msg.data.duration)} | ${msg.data.segments.length} segments`;
                exportActions.classList.remove('opacity-30', 'pointer-events-none');
                compareBanner.classList.remove('hidden');
                btnTranscribe.textContent = `Start Transcription (${chosenModel.toUpperCase()})`;
                renderSpeakerRenameBar(); updateVirtualWindow();
                loadHistory(true);
              } else if (msg.type === 'error') {
                if (timerInterval) clearInterval(timerInterval);
                progressStageText.textContent = '❌ Error: ' + msg.error;
                progressBar.style.width = '100%';
                spinner.classList.add('hidden');
                showToast('Error: ' + msg.error);
              }
            }
          }
        }
      } catch (err) {
        if (timerInterval) clearInterval(timerInterval);
        metaLabel.textContent = 'Interrupted. Click Resume to continue.';
        progressStageText.textContent = '⚠️ Stream interrupted';
        btnResume.classList.remove('hidden');
      } finally {
        streamBadge.classList.add('hidden');
        btnTranscribe.disabled = false;
      }
    }

    let historyCache = [];

    function renderHistoryDOM(sources) {
      if (!sources || !sources.length) {
        historyList.innerHTML = '<div class="text-slate-400 text-center py-6">No history records found.</div>';
        return;
      }
      historyList.innerHTML = '';
      sources.forEach(src => {
        const groupCard = document.createElement('div');
        groupCard.className = 'p-3.5 bg-slate-50 dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2.5';
        
        const modelBadges = src.models.map(m => `<span class="px-2 py-0.5 bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20 rounded-md font-mono text-[10px] font-bold">${m}</span>`).join(' ');
        const canCompare = src.runs.length >= 2;

        let runsHtml = '<div class="space-y-1.5 pt-1">';
        src.runs.forEach(r => {
          const dateStr = new Date(r.created_at * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
          runsHtml += `
            <div onclick="loadHistoryItem('${r.id}')" class="flex items-center justify-between p-2 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-indigo-500 cursor-pointer transition-all">
              <div class="flex items-center gap-2 truncate">
                <span class="font-bold font-mono text-indigo-600 dark:text-indigo-400 text-[11px]">${r.model}</span>
                <span class="text-slate-400 text-[10px]">${dateStr}</span>
                ${r.processing_time ? `<span class="text-[10px] text-emerald-600 dark:text-emerald-400 font-mono">⏱️ ${r.processing_time}s</span>` : ''}
              </div>
              <button onclick="event.stopPropagation(); deleteHistoryRecord('${r.id}')" class="text-slate-400 hover:text-rose-500 p-0.5 rounded transition-colors" title="Delete run">🗑️</button>
            </div>
          `;
        });
        runsHtml += '</div>';

        groupCard.innerHTML = `
          <div class="flex items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-2">
            <div class="truncate font-semibold text-slate-800 dark:text-slate-200 text-xs truncate">
              ${src.source_name}
            </div>
            <div class="flex items-center gap-1.5 flex-shrink-0">
              ${canCompare ? `<button onclick="openCompareWithSource('${encodeURIComponent(src.source_name)}')" class="px-2 py-0.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-[10px] font-bold shadow-sm">⚖️ Compare (${src.runs.length})</button>` : ''}
            </div>
          </div>
          <div class="flex items-center gap-1.5">${modelBadges}</div>
          ${runsHtml}
        `;
        historyList.appendChild(groupCard);
      });
    }

    async function loadHistory(silent = false) {
      if (!appToken) { if (!silent) tokenModal.classList.remove('hidden'); return; }
      if (historyCache.length > 0) {
        renderHistoryDOM(historyCache);
      } else if (!silent) {
        historyList.innerHTML = '<div class="text-slate-400 text-center py-6">Loading history...</div>';
      }
      try {
        const res = await fetch(`/api/sources?token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
        if (res.status === 401) { if (!silent) { tokenGate.classList.remove('hidden'); localStorage.removeItem('appToken'); appToken = ''; } return; }
        const sources = await res.json();
        historyCache = sources;
        compareSourcesData = sources;
        renderHistoryDOM(sources);
      } catch (e) {
        if (!silent && !historyCache.length) historyList.innerHTML = '<div class="text-red-500 text-center py-6">Failed to load history</div>';
      }
    }

    async function deleteHistoryRecord(id) {
      try {
        await fetch(`/api/history/${id}?token=${encodeURIComponent(appToken)}`, { method: 'DELETE', headers: authHeaders() });
        showToast('Run deleted');
        if (currentJobId === id) clearCurrentTimeline();
        loadHistory();
      } catch (e) { alert('Failed to delete: ' + e.message); }
    }

    async function clearAllHistoryRecords() {
      if (!confirm('Are you sure you want to clear all transcription history?')) return;
      try {
        await fetch(`/api/history?token=${encodeURIComponent(appToken)}`, { method: 'DELETE', headers: authHeaders() });
        showToast('All history cleared');
        loadHistory();
      } catch (e) { alert('Failed to clear history: ' + e.message); }
    }

    async function loadHistoryItem(id) {
      historyModal.classList.add('hidden');
      try {
        const res = await fetch(`/api/history/${id}?token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
        const item = await res.json();
        currentResult = item.result; currentJobId = id; currentSourceName = item.source_name || 'transcript';
        const isPaused = item.status === 'in_progress';
        progressBox.classList.remove('hidden'); spinner.classList.add('hidden');
        progressBar.style.width = isPaused ? '50%' : '100%';
        progressStageText.textContent = isPaused ? '⏸️ Paused Checkpoint' : `✔ [${item.model}] Completed Transcription`;
        const dur = currentResult.duration || 0;
        const procInfo = item.processing_time ? ` • ⏱️ Processed in ${item.processing_time}s` : '';
        progressStatus.textContent = `${formatNaturalDuration(dur)} / ${formatNaturalDuration(dur)} (${isPaused ? 'Paused' : '100%'}) • ${currentResult.segments.length} segments${procInfo}`;
        if (isPaused) btnResume.classList.remove('hidden'); else btnResume.classList.add('hidden');
        metaLabel.textContent = `Model: ${item.model.toUpperCase()} | ${item.source_name} | ${currentResult.language.toUpperCase()} | ${formatNaturalDuration(currentResult.duration)}`;
        exportActions.classList.remove('opacity-30', 'pointer-events-none');
        compareBanner.classList.remove('hidden');
        btnTranscribe.disabled = false;
        btnTranscribe.textContent = `Start Transcription (${modelSelect.value.toUpperCase()})`;
        renderSpeakerRenameBar(); updateVirtualWindow();
      } catch (e) { alert('Failed to load item: ' + e.message); }
    }

    /* === MODEL COMPARISON LOGIC === */
    async function openCompareModal() {
      if (!appToken) { tokenModal.classList.remove('hidden'); return; }
      compareModal.classList.remove('hidden');
      await refreshCompareSources();
    }

    function openCompareWithSource(encodedSrc) {
      historyModal.classList.add('hidden');
      compareModal.classList.remove('hidden');
      refreshCompareSources(decodeURIComponent(encodedSrc));
    }

    async function refreshCompareSources(preselectSource = null) {
      try {
        const res = await fetch(`/api/sources?token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
        compareSourcesData = await res.json();
        compareSourceSelect.innerHTML = '';
        if (!compareSourcesData.length) {
          compareSourceSelect.innerHTML = '<option value="">No completed transcriptions available</option>';
          return;
        }

        compareSourcesData.forEach(s => {
          const opt = document.createElement('option');
          opt.value = s.source_name;
          opt.textContent = `${s.source_name} (${s.runs.length} model run${s.runs.length > 1 ? 's' : ''})`;
          compareSourceSelect.appendChild(opt);
        });

        if (preselectSource && compareSourcesData.some(s => s.source_name === preselectSource)) {
          compareSourceSelect.value = preselectSource;
        } else if (currentSourceName && compareSourcesData.some(s => s.source_name === currentSourceName)) {
          compareSourceSelect.value = currentSourceName;
        }

        onCompareSourceChanged();
      } catch (e) {
        showToast('Failed to load comparison sources');
      }
    }

    compareSourceSelect.onchange = () => onCompareSourceChanged();

    function onCompareSourceChanged() {
      const srcName = compareSourceSelect.value;
      const srcObj = compareSourcesData.find(s => s.source_name === srcName);
      if (!srcObj || !srcObj.runs.length) {
        compareRunA.innerHTML = '<option value="">No runs</option>';
        compareRunB.innerHTML = '<option value="">No runs</option>';
        return;
      }

      compareRunA.innerHTML = '';
      compareRunB.innerHTML = '';

      srcObj.runs.forEach((r, idx) => {
        const dateStr = new Date(r.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const text = `${r.model.toUpperCase()} (${dateStr}${r.processing_time ? ` • ${r.processing_time}s` : ''})`;
        
        const optA = document.createElement('option');
        optA.value = r.id;
        optA.textContent = text;
        compareRunA.appendChild(optA);

        const optB = document.createElement('option');
        optB.value = r.id;
        optB.textContent = text;
        compareRunB.appendChild(optB);
      });

      if (srcObj.runs.length >= 2) {
        compareRunA.selectedIndex = 0;
        compareRunB.selectedIndex = 1;
      }

      executeComparison();
    }

    compareRunA.onchange = () => executeComparison();
    compareRunB.onchange = () => executeComparison();

    async function executeComparison() {
      const idA = compareRunA.value;
      const idB = compareRunB.value;
      if (!idA || !idB) return;

      try {
        const res = await fetch(`/api/compare?job_a=${encodeURIComponent(idA)}&job_b=${encodeURIComponent(idB)}&token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
        if (!res.ok) throw new Error('Comparison failed');
        const data = await res.json();
        renderComparisonResults(data);
      } catch (e) {
        console.error(e);
      }
    }

    function renderComparisonResults(data) {
      const { run_a, run_b, similarity_score } = data;

      document.getElementById('metric-similarity').textContent = `${similarity_score}%`;
      document.getElementById('metric-speed').textContent = `A: ${run_a.processing_time}s (${run_a.speedup}x) | B: ${run_b.processing_time}s (${run_b.speedup}x)`;
      
      const fasterModel = run_a.processing_time < run_b.processing_time ? run_a.model : run_b.model;
      const ratio = (run_a.processing_time > 0 && run_b.processing_time > 0) ? (Math.max(run_a.processing_time, run_b.processing_time) / Math.min(run_a.processing_time, run_b.processing_time)).toFixed(1) : '1.0';
      document.getElementById('metric-speedup').textContent = `${fasterModel} is ${ratio}x faster`;

      document.getElementById('metric-words').textContent = `A: ${run_a.word_count} vs B: ${run_b.word_count}`;
      const diffWords = Math.abs(run_a.word_count - run_b.word_count);
      document.getElementById('metric-word-diff').textContent = `${diffWords} word delta`;

      document.getElementById('metric-speakers').textContent = `A: ${run_a.speakers_count} vs B: ${run_b.speakers_count}`;

      document.getElementById('label-model-a').textContent = `Model A: ${run_a.model.toUpperCase()}`;
      document.getElementById('badge-meta-a').textContent = `${run_a.word_count} words • ${run_a.processing_time}s • Lang: ${run_a.language.toUpperCase()}`;

      document.getElementById('label-model-b').textContent = `Model B: ${run_b.model.toUpperCase()}`;
      document.getElementById('badge-meta-b').textContent = `${run_b.word_count} words • ${run_b.processing_time}s • Lang: ${run_b.language.toUpperCase()}`;

      const containerA = document.getElementById('diff-text-a');
      const containerB = document.getElementById('diff-text-b');
      containerA.innerHTML = '';
      containerB.innerHTML = '';

      run_a.diff_words.forEach(w => {
        const span = document.createElement('span');
        span.textContent = w.word + ' ';
        if (w.status === 'deleted') {
          span.className = 'bg-rose-500/20 text-rose-700 dark:text-rose-300 px-1 py-0.5 rounded font-semibold line-through';
        } else if (w.status === 'replaced') {
          span.className = 'bg-amber-500/20 text-amber-700 dark:text-amber-300 px-1 py-0.5 rounded font-semibold';
        } else {
          span.className = 'text-slate-800 dark:text-slate-200';
        }
        containerA.appendChild(span);
      });

      run_b.diff_words.forEach(w => {
        const span = document.createElement('span');
        span.textContent = w.word + ' ';
        if (w.status === 'inserted') {
          span.className = 'bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 px-1 py-0.5 rounded font-semibold underline';
        } else if (w.status === 'replaced') {
          span.className = 'bg-amber-500/20 text-amber-700 dark:text-amber-300 px-1 py-0.5 rounded font-semibold';
        } else {
          span.className = 'text-slate-800 dark:text-slate-200';
        }
        containerB.appendChild(span);
      });

      renderComparativeTimeline(run_a, run_b);
    }

    function renderComparativeTimeline(run_a, run_b) {
      const container = document.getElementById('timeline-comparative-container');
      container.innerHTML = '';

      const maxSegs = Math.max(run_a.segments.length, run_b.segments.length);
      for (let i = 0; i < maxSegs; i++) {
        const sa = run_a.segments[i] || null;
        const sb = run_b.segments[i] || null;

        const row = document.createElement('div');
        row.className = 'p-3 bg-slate-50 dark:bg-slate-950/60 rounded-xl border border-slate-200 dark:border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4';

        const htmlA = sa ? `
          <div class="space-y-1">
            <div class="flex items-center justify-between text-[11px] text-slate-500">
              <span class="font-bold text-indigo-600 dark:text-indigo-400">[${run_a.model.toUpperCase()}] ${sa.speaker || 'Speaker 1'}</span>
              <span class="font-mono">${formatTime(sa.start)} → ${formatTime(sa.end)}</span>
            </div>
            <p class="text-slate-800 dark:text-slate-200">${sa.text}</p>
          </div>
        ` : `<div class="text-slate-400 italic">No corresponding segment</div>`;

        const htmlB = sb ? `
          <div class="space-y-1 border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-800 pt-2 md:pt-0 md:pl-4">
            <div class="flex items-center justify-between text-[11px] text-slate-500">
              <span class="font-bold text-purple-600 dark:text-purple-400">[${run_b.model.toUpperCase()}] ${sb.speaker || 'Speaker 1'}</span>
              <span class="font-mono">${formatTime(sb.start)} → ${formatTime(sb.end)}</span>
            </div>
            <p class="text-slate-800 dark:text-slate-200">${sb.text}</p>
          </div>
        ` : `<div class="text-slate-400 italic md:border-l border-slate-200 dark:border-slate-800 md:pl-4">No corresponding segment</div>`;

        row.innerHTML = htmlA + htmlB;
        container.appendChild(row);
      }
    }

    function toggleCompareView() {
      const splitView = document.getElementById('compare-split-view');
      const timelineView = document.getElementById('compare-timeline-view');
      const btn = document.getElementById('btn-toggle-view');

      if (compareViewMode === 'diff') {
        compareViewMode = 'timeline';
        splitView.classList.add('hidden');
        timelineView.classList.remove('hidden');
        btn.textContent = 'Switch to Word Diff View';
      } else {
        compareViewMode = 'diff';
        timelineView.classList.add('hidden');
        splitView.classList.remove('hidden');
        btn.textContent = 'Switch to Timeline View';
      }
    }

    function getAudioStem() {
      if (!currentSourceName) return 'transcript';
      let stem = currentSourceName;
      const gdMatch = stem.match(/\/d\/([a-zA-Z0-9_-]+)/) || stem.match(/id=([a-zA-Z0-9_-]+)/);
      if (gdMatch) return `gdrive_${gdMatch[1]}`;
      const lastSlash = stem.lastIndexOf('/'); if (lastSlash !== -1) stem = stem.substring(lastSlash + 1);
      const lastDot = stem.lastIndexOf('.'); if (lastDot !== -1) stem = stem.substring(0, lastDot);
      return stem.replace(/[^a-zA-Z0-9_-]/g, '_').substring(0, 64) || 'transcript';
    }

    function formatNaturalDuration(s) {
      const secNum = Math.floor(s || 0);
      const hrs = Math.floor(secNum / 3600), mins = Math.floor((secNum % 3600) / 60), secs = secNum % 60;
      if (hrs > 0) return `${hrs}h ${mins}m ${secs.toString().padStart(2, '0')}s`;
      if (mins > 0) return `${mins}m ${secs.toString().padStart(2, '0')}s`;
      return `${secs}s`;
    }

    function generateFormattedContent(style = 'full', isMarkdown = true) {
      if (!currentResult || !currentResult.segments.length) return '';
      const segs = currentResult.segments;
      if (style === 'text_only') return segs.map(s => s.text.trim()).join(isMarkdown ? '\n\n' : '\n');
      if (style === 'no_ts') {
        if (isMarkdown) return segs.map(s => `> **${s.speaker || 'Speaker 1'}**:\n> ${s.text}`).join('\n\n');
        return segs.map(s => `[${s.speaker || 'Speaker 1'}]: ${s.text}`).join('\n');
      }
      if (style === 'speaker_only') {
        const groups = {};
        segs.forEach(s => { const spk = s.speaker || 'Speaker 1'; if (!groups[spk]) groups[spk] = []; groups[spk].push(s.text); });
        if (isMarkdown) return Object.keys(groups).map(spk => `### 👤 ${spk}\n` + groups[spk].map(t => `- ${t}`).join('\n')).join('\n\n');
        return Object.keys(groups).map(spk => `=== ${spk} ===\n` + groups[spk].join('\n')).join('\n\n');
      }
      const speakers = (currentResult.speakers || ['Speaker 1']).join(', ');
      if (isMarkdown) {
        let md = `# 🎙️ Audio Transcription Transcript\n\n| Property | Details |\n| :--- | :--- |\n| **Audio Source** | ${currentSourceName} |\n| **Duration** | ${formatNaturalDuration(currentResult.duration)} (${currentResult.duration.toFixed(1)}s) |\n| **Language** | ${currentResult.language.toUpperCase()} |\n| **Speakers** | ${speakers} |\n| **Segments** | ${segs.length} |\n\n---\n\n### 💬 Dialogue Timeline\n\n`;
        segs.forEach(s => { md += `> **[${formatTime(s.start)} ➜ ${formatTime(s.end)}] ${s.speaker || 'Speaker 1'}**:\n> ${s.text}\n\n`; });
        return md;
      }
      return segs.map(s => `[${s.speaker || 'Speaker 1'}] (${formatTime(s.start)} - ${formatTime(s.end)}): ${s.text}`).join('\n');
    }

    function copyMarkdown() {
      const style = exportStyle.value;
      const content = generateFormattedContent(style, true);
      if (!content) return;
      navigator.clipboard.writeText(content).then(() => showToast(`Copied (${exportStyle.options[exportStyle.selectedIndex].text})!`));
    }

    function formatTime(s) {
      const secNum = Math.floor(s || 0);
      const hrs = Math.floor(secNum / 3600), mins = Math.floor((secNum % 3600) / 60), secs = secNum % 60;
      if (hrs > 0) return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
      return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function downloadFormat(fmt) {
      if (!currentResult || !currentResult.segments.length) return;
      const stem = getAudioStem(), style = exportStyle.value;
      let text = '', mime = 'text/plain', filename = `${stem}.${fmt}`;
      if (fmt === 'md') { text = generateFormattedContent(style, true); mime = 'text/markdown'; }
      else if (fmt === 'txt') { text = generateFormattedContent(style, false); }
      else if (fmt === 'json') { text = JSON.stringify(currentResult, null, 2); mime = 'application/json'; }
      else if (fmt === 'srt') {
        text = currentResult.segments.map((s, i) => {
          const spkTag = (style === 'text_only') ? '' : `[${s.speaker || 'Speaker 1'}] `;
          return `${i+1}\n${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n${spkTag}${s.text}\n`;
        }).join('\n');
      } else if (fmt === 'vtt') {
        text = 'WEBVTT\n\n' + currentResult.segments.map(s => {
          const spkTag = (style === 'text_only') ? '' : `<v ${s.speaker || 'Speaker 1'}>`;
          return `${formatVttTime(s.start)} --> ${formatVttTime(s.end)}\n${spkTag}${s.text}\n`;
        }).join('\n');
      }
      const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: mime })); a.download = filename; a.click();
    }
    function formatSrtTime(s) { const hrs = Math.floor(s / 3600).toString().padStart(2, '0'), mins = Math.floor((s % 3600) / 60).toString().padStart(2, '0'), secs = Math.floor(s % 60).toString().padStart(2, '0'), ms = Math.floor((s - Math.floor(s)) * 1000).toString().padStart(3, '0'); return `${hrs}:${mins}:${secs},${ms}`; }
    function formatVttTime(s) { return formatSrtTime(s).replace(',', '.'); }

    let currentMomMarkdown = '';

    async function openMomModal() {
      const modal = document.getElementById('mom-modal');
      const content = document.getElementById('mom-content');
      modal.classList.remove('hidden');

      if (!currentResult || !currentResult.segments || !currentResult.segments.length) {
        content.innerText = 'No transcription segments available to generate MOM.';
        return;
      }

      if (currentJobId) {
        try {
          const res = await fetch(`/api/history/${currentJobId}/mom?token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
          if (res.ok) {
            const data = await res.json();
            if (data.mom_markdown) {
              currentMomMarkdown = data.mom_markdown;
              content.innerText = currentMomMarkdown;
              return;
            }
          }
        } catch (e) {}
      }

      await generateMomStream();
    }

    function closeMomModal() {
      document.getElementById('mom-modal').classList.add('hidden');
    }

    async function generateMomStream() {
      const content = document.getElementById('mom-content');
      const status = document.getElementById('mom-status');
      status.classList.remove('hidden');
      content.innerText = '';
      currentMomMarkdown = '';

      try {
        const response = await fetch(`/api/mom?token=${encodeURIComponent(appToken)}`, {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            job_id: currentJobId,
            segments: currentResult.segments,
            stream: true,
          })
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(err.detail || 'Failed to generate MOM');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.chunk) {
                currentMomMarkdown += parsed.chunk;
                content.innerText = currentMomMarkdown;
                content.scrollTop = content.scrollHeight;
              }
            } catch (e) {}
          }
        }
      } catch (err) {
        content.innerText = `⚠️ Error generating MOM: ${err.message}\n\nPlease verify that FreeToken server is running on http://127.0.0.1:4012 (or configure LLM_BASE_URL in .env).`;
      } finally {
        status.classList.add('hidden');
      }
    }

    function copyMomMarkdown() {
      if (!currentMomMarkdown) return;
      navigator.clipboard.writeText(currentMomMarkdown).then(() => showToast('MOM Markdown copied to clipboard!'));
    }

    function downloadMomMarkdown() {
      if (!currentMomMarkdown) return;
      const stem = getAudioStem();
      const filename = `${stem}.mom.md`;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([currentMomMarkdown], { type: 'text/markdown' }));
      a.download = filename;
      a.click();
    }

    let currentRefinedText = '';

    async function openRefineModal() {
      const modal = document.getElementById('refine-modal');
      const content = document.getElementById('refine-content');
      modal.classList.remove('hidden');

      if (!currentResult || !currentResult.segments || !currentResult.segments.length) {
        content.innerText = 'No transcription segments available to refine.';
        return;
      }

      if (currentJobId) {
        try {
          const res = await fetch(`/api/history/${currentJobId}/refined?token=${encodeURIComponent(appToken)}`, { headers: authHeaders() });
          if (res.ok) {
            const data = await res.json();
            if (data.refined_text) {
              currentRefinedText = data.refined_text;
              content.innerText = currentRefinedText;
              return;
            }
          }
        } catch (e) {}
      }

      await refineTranscriptStream();
    }

    function closeRefineModal() {
      document.getElementById('refine-modal').classList.add('hidden');
    }

    async function refineTranscriptStream() {
      const content = document.getElementById('refine-content');
      const status = document.getElementById('refine-status');
      status.classList.remove('hidden');
      content.innerText = '';
      currentRefinedText = '';

      try {
        const response = await fetch(`/api/refine?token=${encodeURIComponent(appToken)}`, {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            job_id: currentJobId,
            segments: currentResult.segments,
            stream: true,
          })
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(err.detail || 'Failed to refine transcript');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.chunk) {
                currentRefinedText += parsed.chunk;
                content.innerText = currentRefinedText;
                content.scrollTop = content.scrollHeight;
              }
            } catch (e) {}
          }
        }
      } catch (err) {
        content.innerText = `⚠️ Error refining transcript: ${err.message}\n\nPlease verify that FreeToken server is running on http://127.0.0.1:4012 (or configure LLM_BASE_URL in .env).`;
      } finally {
        status.classList.add('hidden');
      }
    }

    function copyRefinedText() {
      if (!currentRefinedText) return;
      navigator.clipboard.writeText(currentRefinedText).then(() => showToast('Refined transcript copied to clipboard!'));
    }

    function downloadRefinedText() {
      if (!currentRefinedText) return;
      const stem = getAudioStem();
      const filename = `${stem}.refined.txt`;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([currentRefinedText], { type: 'text/plain' }));
      a.download = filename;
      a.click();
    }
  </script>
</body>
</html>
"""
