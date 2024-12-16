// UI related functions
const UI = {
    elements: {
        form: document.getElementById('searchForm'),
        input: document.getElementById('nameInput'),
        submitButton: document.querySelector('button[type="submit"]'),
        errorMessage: document.getElementById('errorMessage'),
        resultsList: document.getElementById('resultsList'),
        searchIcon: document.querySelector('.search-icon'),
        loader: document.querySelector('.loader')
    },

    showLoading() {
        this.elements.submitButton.disabled = true;
        this.elements.searchIcon.classList.add('hidden');
        this.elements.loader.classList.remove('hidden');
    },

    hideLoading() {
        this.elements.submitButton.disabled = false;
        this.elements.searchIcon.classList.remove('hidden');
        this.elements.loader.classList.add('hidden');
    },

    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorMessage.classList.remove('hidden');
    },

    hideError() {
        this.elements.errorMessage.classList.add('hidden');
    },

    displayResults(names) {
        this.elements.resultsList.innerHTML = '';
        
        if (names.length === 0) {
            this.elements.resultsList.innerHTML = '<p class="empty-message">No similar names found</p>';
            return;
        }

        names.forEach(name => {
            const item = document.createElement('div');
            item.className = 'result-item';
            item.textContent = name;
            this.elements.resultsList.appendChild(item);
        });
    },

    clearResults() {
        this.elements.resultsList.innerHTML = '<p class="empty-message">Enter a name to see results</p>';
    }
};