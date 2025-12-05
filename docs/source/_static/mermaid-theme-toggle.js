/**
 * Mermaid Theme Toggle for pydata-sphinx-theme + sphinxcontrib-mermaid
 * 
 * Re-renders Mermaid diagrams when the site theme changes.
 * Compatible with file:// protocol (no ES module imports).
 */

(function() {
    'use strict';
    
    // Store for original diagram sources (keyed by index)
    var DIAGRAM_SOURCES = {};
    var mermaidReady = false;
    
    /**
     * Capture Mermaid diagram sources before they get rendered to SVG
     */
    function captureSources() {
        var diagrams = document.querySelectorAll('pre.mermaid');
        diagrams.forEach(function(pre, idx) {
            // Only capture if not already captured and not yet rendered (no SVG)
            if (!DIAGRAM_SOURCES[idx] && !pre.querySelector('svg')) {
                var source = pre.textContent.trim();
                if (source) {
                    DIAGRAM_SOURCES[idx] = source;
                    pre.setAttribute('data-diagram-idx', idx);
                }
            }
        });
        console.log('[mermaid-theme-toggle] Captured ' + Object.keys(DIAGRAM_SOURCES).length + ' diagram sources');
    }
    
    /**
     * Get current theme from pydata-sphinx-theme
     */
    function getCurrentTheme() {
        var dataTheme = document.documentElement.dataset.theme;
        if (!dataTheme || dataTheme === 'auto') {
            return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        return dataTheme;
    }
    
    /**
     * Map site theme to Mermaid theme
     */
    function getMermaidTheme() {
        return getCurrentTheme() === 'dark' ? 'dark' : 'default';
    }
    
    /**
     * Wait for mermaid to be available globally
     */
    function waitForMermaid(callback, maxAttempts) {
        maxAttempts = maxAttempts || 50;
        var attempts = 0;
        
        var check = function() {
            attempts++;
            if (typeof mermaid !== 'undefined' && mermaid.initialize) {
                callback(mermaid);
            } else if (attempts < maxAttempts) {
                setTimeout(check, 100);
            } else {
                console.warn('[mermaid-theme-toggle] Mermaid not found after ' + maxAttempts + ' attempts');
            }
        };
        check();
    }
    
    /**
     * Re-render all diagrams with current theme
     */
    function rerenderDiagrams() {
        waitForMermaid(function(mermaid) {
            var theme = getMermaidTheme();
            console.log('[mermaid-theme-toggle] Re-rendering with theme: ' + theme);
            
            mermaid.initialize({
                startOnLoad: false,
                theme: theme,
                securityLevel: 'loose',
                flowchart: { useMaxWidth: true, htmlLabels: true }
            });
            
            var indices = Object.keys(DIAGRAM_SOURCES);
            var renderNext = function(i) {
                if (i >= indices.length) return;
                
                var idx = indices[i];
                var source = DIAGRAM_SOURCES[idx];
                var pre = document.querySelector('pre.mermaid[data-diagram-idx="' + idx + '"]');
                
                if (!pre || !source) {
                    renderNext(i + 1);
                    return;
                }
                
                var id = 'mermaid-theme-' + Date.now() + '-' + idx;
                
                mermaid.render(id, source).then(function(result) {
                    pre.innerHTML = result.svg;
                    
                    // Re-apply d3 zoom if available
                    if (typeof d3 !== 'undefined' && d3.select && d3.zoom) {
                        var svgEl = pre.querySelector('svg');
                        if (svgEl) {
                            var svgD3 = d3.select(svgEl);
                            if (!svgEl.querySelector('g.wrapper')) {
                                svgD3.html("<g class='wrapper'>" + svgD3.html() + "</g>");
                            }
                            var inner = svgD3.select("g.wrapper");
                            var zoom = d3.zoom().on("zoom", function(event) {
                                inner.attr("transform", event.transform);
                            });
                            svgD3.call(zoom);
                        }
                    }
                    
                    renderNext(i + 1);
                }).catch(function(err) {
                    console.warn('[mermaid-theme-toggle] Failed to render diagram ' + idx + ':', err);
                    renderNext(i + 1);
                });
            };
            
            renderNext(0);
        });
    }
    
    /**
     * Set up theme change observer
     */
    function setupObserver() {
        // Watch for theme attribute changes on <html>
        var observer = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                if (mutations[i].attributeName === 'data-theme') {
                    console.log('[mermaid-theme-toggle] Theme changed, re-rendering...');
                    rerenderDiagrams();
                    break;
                }
            }
        });
        
        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
        
        // Watch for system preference changes when in 'auto' mode
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
            var theme = document.documentElement.dataset.theme;
            if (!theme || theme === 'auto') {
                console.log('[mermaid-theme-toggle] System preference changed, re-rendering...');
                rerenderDiagrams();
            }
        });
        
        console.log('[mermaid-theme-toggle] Observer set up, watching for theme changes');
    }
    
    /**
     * Override theme button to only toggle between light and dark (skip auto)
     */
    function setupBinaryThemeToggle() {
        // Find all theme switch buttons (there may be multiple in responsive layouts)
        var buttons = document.querySelectorAll('.theme-switch-button');
        
        buttons.forEach(function(btn) {
            // Clone and replace to remove existing event listeners
            var newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            newBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                // Get current effective theme (resolving 'auto' to actual preference)
                var currentTheme = getCurrentTheme();
                // Toggle to opposite
                var newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                
                // Set the theme - PST uses both 'mode' and 'theme' in localStorage
                document.documentElement.dataset.theme = newTheme;
                document.documentElement.dataset.mode = newTheme;
                localStorage.setItem('theme', newTheme);
                localStorage.setItem('mode', newTheme);
                
                console.log('[mermaid-theme-toggle] Toggled theme: ' + currentTheme + ' → ' + newTheme);
            });
        });
        
        console.log('[mermaid-theme-toggle] Binary theme toggle set up (no auto mode in button)');
    }
    
    // Capture sources at DOMContentLoaded (before Mermaid's 'load' handler renders them)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', captureSources);
    } else {
        captureSources();
    }
    
    // Set up observer after page fully loads
    window.addEventListener('load', function() {
        // Final attempt to capture any sources we might have missed
        captureSources();
        // Set up the theme change observer
        setupObserver();
        // Override theme button to only toggle light/dark (skip auto)
        setupBinaryThemeToggle();
    });
    
})();
