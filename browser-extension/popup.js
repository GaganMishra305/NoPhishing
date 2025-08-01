// extension/popup.js

document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('modelSelect');
    const saveButton = document.getElementById('saveButton');
    const statusMessage = document.getElementById('statusMessage');

    // Load saved preference when popup opens
    // Use 'browser' API for Firefox compatibility, fallback to 'chrome' for Chrome
    const browserApi = typeof browser !== 'undefined' ? browser : chrome;

    browserApi.storage.local.get(['selectedModel'], (result) => {
        if (result.selectedModel) {
            modelSelect.value = result.selectedModel;
        } else {
            // Default to ANN if no preference is saved
            modelSelect.value = 'ANN'; 
        }
    });

    // Save preference when button is clicked
    saveButton.addEventListener('click', () => {
        const selectedModel = modelSelect.value;
        browserApi.storage.local.set({ selectedModel: selectedModel }, () => {
            statusMessage.textContent = 'Setting saved!';
            setTimeout(() => {
                statusMessage.textContent = '';
            }, 2000);

            // Send message to all content scripts to update their model
            browserApi.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                if (tabs[0]) {
                    browserApi.tabs.sendMessage(tabs[0].id, {
                        action: 'updateModel',
                        model: selectedModel
                    }, (response) => {
                        // Check for runtime.lastError for both Chrome and Firefox
                        if (browserApi.runtime.lastError) {
                            console.warn("Phishing Detector: Could not send message to content script:", browserApi.runtime.lastError.message);
                            statusMessage.textContent = 'Setting saved (page reload might be needed)!';
                        } else if (response && response.status === 'success') {
                            console.log('Phishing Detector: Content script updated successfully.');
                        }
                    });
                }
            });
        });
    });
});
