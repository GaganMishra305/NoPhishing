// extension/content.js
    
// --- Configuration ---
// URL of your Python Flask API endpoint for prediction
// IMPORTANT: This should be reachable from the browser. If running Docker, use localhost or the container's IP.
const API_PREDICT_URL = "http://127.0.0.1:5000/predict"; 
// Time in milliseconds before sending a request after hover (to prevent too many requests)
const HOVER_DELAY_MS = 500;
// Timeout for the API request
const API_TIMEOUT_MS = 5000; // 5 seconds
// --- End Configuration ---
    
let tooltipContainer = null; // Main container element for the tooltip
let activeTimeout = null;    // Stores the timeout ID for delayed hover
let currentLink = null;      // Stores the currently hovered link element
let abortController = null;  // Used to abort fetch requests if link changes quickly
    
// Cache for predictions to avoid re-querying the API for the same URL and model
const predictionCache = new Map();

// --- Model Selection State (managed by popup.js and stored in Chrome Storage) ---
let currentApiModelArchitecture = 'ANN'; // Default model on initial load
let currentApiEnsembleFlag = false;     // Default to not using ensemble on initial load

/**
 * Maps a phishing probability (0 to 1) to an HSL color for a green-to-red gradient.
 * 0 (0% phishing) -> Green (Hue=120)
 * 1 (100% phishing) -> Red (Hue=0)
 * @param {number} probability A number between 0 and 1 representing phishing probability.
 * @returns {string} An HSL color string (e.g., "hsl(120, 80%, 45%)").
 */
function getPhishingColor(probability) {
    // Clamp probability between 0 and 1 to ensure valid color range
    probability = Math.max(0, Math.min(1, probability));
    
    // Hue ranges from Green (120) to Red (0) as probability goes from 0 to 1
    const hue = (1 - probability) * 120; 
    const saturation = '80%';
    const lightness = '45%'; // Adjusted for better text contrast
    
    return `hsl(${hue}, ${saturation}, ${lightness})`;
}

/**
 * Creates and appends the main tooltip container to the document body if it doesn't already exist.
 * This container holds the actual tooltip message.
 * @returns {HTMLElement} The tooltip container element.
 */
function getOrCreateTooltipContainer() {
    if (!tooltipContainer) {
        tooltipContainer = document.createElement('div');
        tooltipContainer.id = 'phishing-detector-tooltip-container';
        document.body.appendChild(tooltipContainer);
    }
    return tooltipContainer;
}

/**
 * Displays a loading spinner message within the tooltip.
 * This is shown immediately upon hovering a link.
 * @param {HTMLElement} linkElement The link element currently being hovered over.
 */
function showLoadingTooltip(linkElement) {
    const container = getOrCreateTooltipContainer();
    const tooltip = document.createElement('div');
    tooltip.className = 'phishing-detector-tooltip phishing-detector-loading'; // Apply loading styles
    tooltip.innerHTML = 'Loading...'; // Text for the loading state
    container.innerHTML = ''; // Clear any previous tooltip content
    container.appendChild(tooltip); // Add the new loading tooltip
    positionTooltip(tooltip, linkElement); // Position it relative to the link
    // RequestAnimationFrame ensures the DOM is ready for the CSS transition to apply
    requestAnimationFrame(() => tooltip.classList.add('show'));
}

/**
 * Displays the prediction result (Legitimate/Phishing and confidence) in the tooltip.
 * The tooltip's color dynamically changes based on the phishing probability.
 * @param {string} predictionLabel 'legitimate' or 'phishing'.
 * @param {number} confidence The confidence score (0-1) provided by the API.
 * @param {HTMLElement} linkElement The link element being hovered over.
 */
function showPredictionTooltip(predictionLabel, confidence, linkElement) {
    const container = getOrCreateTooltipContainer();
    const tooltip = document.createElement('div');
    
    // Determine the raw phishing probability to calculate the color gradient.
    // If the prediction is 'phishing', the confidence is already the phishing probability.
    // If the prediction is 'legitimate', the phishing probability is (1 - confidence).
    let phishingProbability = 0;
    if (predictionLabel.toLowerCase() === 'phishing') {
        phishingProbability = confidence;
    } else {
        phishingProbability = 1 - confidence; 
    }

    // Get the background color based on the phishing probability
    const bgColor = getPhishingColor(phishingProbability);
    // Calculate a slightly darker border color for visual contrast
    const borderColor = `hsl(${getPhishingColor(phishingProbability).split('(')[1].split(',')[0]}, 70%, 35%)`; 
    
    // Format confidence as a percentage string
    const percentage = (confidence * 100).toFixed(2) + '%';

    // Set dynamic styles and content
    tooltip.className = `phishing-detector-tooltip`; // Base class
    tooltip.style.backgroundColor = bgColor;
    tooltip.style.borderColor = borderColor;
    tooltip.style.color = 'white'; // Keep text white for readability on varied colored backgrounds
    tooltip.innerHTML = `
        <span class="phishing-detector-label">${predictionLabel.charAt(0).toUpperCase() + predictionLabel.slice(1)}</span>
        <span class="phishing-detector-percentage">${percentage}</span>
    `;
    
    container.innerHTML = ''; // Clear previous content (e.g., loading spinner)
    container.appendChild(tooltip); // Add the new prediction tooltip
    positionTooltip(tooltip, linkElement); // Position it correctly
    requestAnimationFrame(() => tooltip.classList.add('show')); // Show with transition
}

/**
 * Positions the tooltip element near the hovered link, adjusting for screen boundaries.
 * @param {HTMLElement} tooltipElement The tooltip div to position.
 * @param {HTMLElement} linkElement The link element that triggered the tooltip.
 */
function positionTooltip(tooltipElement, linkElement) {
    const linkRect = linkElement.getBoundingClientRect(); // Get link's size and position
    const tooltipRect = tooltipElement.getBoundingClientRect(); // Get tooltip's size

    // Initial position: 5px below and aligned with the left edge of the link
    let top = window.scrollY + linkRect.bottom + 5; 
    let left = window.scrollX + linkRect.left;

    // Adjust if tooltip goes off screen to the right
    if (left + tooltipRect.width > window.innerWidth + window.scrollX - 10) { // 10px right margin
        left = window.scrollX + window.innerWidth - tooltipRect.width - 10;
    }
    // Adjust if tooltip goes off screen to the left
    if (left < window.scrollX + 10) { // 10px left margin
        left = window.scrollX + 10;
    }

    // Adjust if tooltip goes off screen at the bottom, move it above the link instead
    if (top + tooltipRect.height > window.innerHeight + window.scrollY - 10) { // 10px bottom margin
        top = window.scrollY + linkRect.top - tooltipRect.height - 5; // 5px above the link
        // If it still goes off screen at the top, place it at the link's Y position as a last resort
        if (top < window.scrollY + 10) {
            top = window.scrollY + linkRect.top;
        }
    }

    tooltipContainer.style.top = `${top}px`;
    tooltipContainer.style.left = `${left}px`;
}

/**
 * Hides the currently displayed tooltip, applying a fade-out transition.
 */
function hideTooltip() {
    if (tooltipContainer) {
        const tooltip = tooltipContainer.querySelector('.phishing-detector-tooltip');
        if (tooltip) {
            tooltip.classList.remove('show'); // Trigger CSS fade-out
            // Remove the tooltip element from the DOM after its transition completes
            setTimeout(() => {
                if (tooltipContainer) { // Ensure container still exists before clearing
                    tooltipContainer.innerHTML = '';
                }
            }, 200); // Matches CSS transition duration
        }
    }
}

/**
 * Makes an API call to your Flask backend to get the phishing prediction for a URL.
 * Includes the currently selected model architecture and ensemble flag in the request.
 * @param {string} url The URL to send to the API for prediction.
 * @returns {Promise<object|null>} A Promise that resolves to the prediction data (object with prediction, confidence)
 * or null if the API call fails or is aborted.
 */
async function getPredictionFromAPI(url) {
    // Create a unique cache key based on the URL, model architecture, and ensemble status.
    const cacheKey = `${currentApiModelArchitecture}-${currentApiEnsembleFlag}-${url}`;

    // Return cached prediction if available
    if (predictionCache.has(cacheKey)) {
        console.log(`Phishing Detector: Using cached prediction for ${url} (${currentApiModelArchitecture}, Ensemble: ${currentApiEnsembleFlag})`);
        return predictionCache.get(cacheKey);
    }

    // Abort any pending API requests from previous quick hovers
    if (abortController) {
        abortController.abort();
    }
    abortController = new AbortController(); // Create a new AbortController for this request
    const signal = abortController.signal; // Get its signal

    try {
        // Create a promise that will reject if the API request takes too long
        const timeoutPromise = new Promise((resolve, reject) => 
            setTimeout(() => reject(new Error('API request timed out')), API_TIMEOUT_MS)
        );

        // Construct the request body with URL and model selection
        const requestBody = { 
            url: url, 
            architecture: currentApiModelArchitecture // Send the selected model name
        };
        if (currentApiEnsembleFlag) {
            requestBody.ensemble = true; // Add ensemble flag if selected
        }

        // Send the POST request to your Flask API
        const fetchPromise = fetch(API_PREDICT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: signal // Attach the abort signal to the fetch request
        });

        // Race the fetch request against the timeout promise
        const response = await Promise.race([fetchPromise, timeoutPromise]);
        
        // Check if the HTTP response was successful (status 2xx)
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
        }

        // Parse the JSON response
        const data = await response.json();
        // Validate the structure of the prediction data
        if (data && data.prediction && data.confidence !== undefined) {
            predictionCache.set(cacheKey, data); // Cache the successful prediction
            return data;
        } else {
            throw new Error('Invalid prediction data received from API.');
        }
    } catch (error) {
        // Handle request aborts (e.g., user moved mouse quickly to another link)
        if (error.name === 'AbortError') {
            console.log('Phishing Detector: API request aborted (new hover detected).');
        } else {
            // Log other API request failures
            console.error('Phishing Detector API request failed:', error);
        }
        return null; // Return null on any failure
    } finally {
        abortController = null; // Clear the controller after the request is done or aborted
    }
}

/**
 * Event handler for `mouseover` events on the document.
 * This is the main entry point for detecting link hovers.
 * @param {Event} event The DOM `mouseover` event object.
 */
async function handleMouseOver(event) {
    // Find the closest anchor tag (`<a>`) ancestor of the hovered element
    const linkElement = event.target.closest('a');
    
    // Ignore if not a link, or if it's an internal page anchor or mailto link
    if (!linkElement || !linkElement.href || linkElement.href.startsWith('#') || linkElement.protocol === 'mailto:') {
        return;
    }

    // If already hovering over the same link, do nothing
    if (linkElement === currentLink) {
        return;
    }

    // Clear any previous delayed hover timeouts and hide any old tooltip immediately
    if (activeTimeout) {
        clearTimeout(activeTimeout);
        activeTimeout = null;
    }
    hideTooltip(); 
    
    currentLink = linkElement; // Set the new current hovered link

    showLoadingTooltip(linkElement); // Immediately show a loading spinner

    // Set a delay before sending the actual API request to avoid spamming the API
    activeTimeout = setTimeout(async () => {
        const url = linkElement.href;
        const prediction = await getPredictionFromAPI(url);

        // Crucial check: Ensure we are *still* hovering over the same link
        // This prevents showing outdated predictions if the user moves mouse quickly
        if (linkElement === currentLink) { 
            if (prediction) {
                // Show the prediction result if API call was successful
                showPredictionTooltip(prediction.prediction, prediction.confidence, linkElement);
            } else {
                // Show a generic error message if prediction failed
                const container = getOrCreateTooltipContainer();
                const tooltip = document.createElement('div');
                tooltip.className = 'phishing-detector-tooltip';
                tooltip.style.backgroundColor = '#f0f0f0'; // Neutral grey background
                tooltip.style.borderColor = '#ccc'; // Neutral border
                tooltip.style.color = '#555'; // Dark grey text
                tooltip.innerHTML = 'Could not get prediction.';
                container.innerHTML = '';
                container.appendChild(tooltip);
                positionTooltip(tooltip, linkElement);
                requestAnimationFrame(() => tooltip.classList.add('show'));
            }
        } else {
            // If the link changed during the delay, hide the tooltip (new hover will handle its own display)
            hideTooltip(); 
        }
        activeTimeout = null; // Clear the timeout ID
    }, HOVER_DELAY_MS);
}

/**
 * Event handler for `mouseout` events on the document.
 * Hides the tooltip when the mouse leaves a link.
 */
function handleMouseOut() {
    // Clear any pending delayed hover timeout
    if (activeTimeout) {
        clearTimeout(activeTimeout);
        activeTimeout = null;
    }
    // Abort any pending API request immediately
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    currentLink = null; // Reset the current hovered link
    hideTooltip(); // Hide the tooltip
}

// --- Initial setup and message listener ---

/**
 * Loads the initial model preference from Chrome storage when the content script is first injected
 * or when the browser starts.
 */
chrome.storage.local.get(['selectedModel'], (result) => {
    if (result.selectedModel) {
        updateModelSelection(result.selectedModel);
        console.log(`Phishing Detector: Initial model set to "${result.selectedModel}".`);
    } else {
        // If no preference is saved, set a default and save it
        const defaultModel = 'ANN';
        updateModelSelection(defaultModel); 
        chrome.storage.local.set({ selectedModel: defaultModel });
        console.log(`Phishing Detector: No model preference found, defaulting to "${defaultModel}" and saving.`);
    }
});

/**
 * Listens for messages from the extension's popup script.
 * This is used to update the model selection dynamically without reloading the page.
 * @param {object} request The message object sent from the popup.
 * @param {object} sender Information about the sender of the message.
 * @param {function} sendResponse Function to call to send a response back to the sender.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateModel' && request.model) {
        updateModelSelection(request.model); // Update the active model based on popup's selection
        console.log(`Phishing Detector: Model updated to "${request.model}" by popup.`);
        predictionCache.clear(); // Clear the prediction cache as the model changed
        hideTooltip(); // Hide any active tooltip
        sendResponse({ status: 'success' }); // Acknowledge message receipt
    }
    // For other actions or if not handled, don't sendResponse or return true, as it indicates no interest.
});

/**
 * Updates the global variables for the selected model architecture and ensemble flag.
 * This function determines whether to use a single model or the ensemble based on the `selectedModel` string.
 * @param {string} selectedModel The model string from the popup (e.g., "ANN", "Ensemble", "CNN", etc.).
 */
function updateModelSelection(selectedModel) {
    if (selectedModel.toLowerCase() === 'ensemble') {
        currentApiEnsembleFlag = true;
        currentApiModelArchitecture = ''; // Architecture name is not relevant for ensemble API call
    } else {
        currentApiEnsembleFlag = false;
        currentApiModelArchitecture = selectedModel; // Use the selected model name
    }
    console.log(`Phishing Detector: Active predictor -> Model: "${currentApiModelArchitecture || 'Ensemble'}" (Ensemble Mode: ${currentApiEnsembleFlag})`);
}


// Add event listeners to the entire document.
// Using `true` as the third argument for `addEventListener` means the listener
// will be triggered in the capturing phase. This ensures that the event is
// caught before it bubbles down to specific elements, allowing us to accurately
// identify the 'a' tag even if the mouse is over a child element of the link.
document.addEventListener('mouseover', handleMouseOver, true); 
document.addEventListener('mouseout', handleMouseOut, true);

console.log("Phishing Detector Content Script Loaded.");

