// Main application logic
document.addEventListener('DOMContentLoaded', () => {
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        const name = UI.elements.input.value.trim();
        if (!name) return;

        UI.hideError();
        UI.showLoading();

        try {
            const similarNames = await API.getSimilarNames(name);
            UI.displayResults(similarNames);
        } catch (error) {
            UI.showError(error.message);
            UI.clearResults();
        } finally {
            UI.hideLoading();
        }
    };

    UI.elements.form.addEventListener('submit', handleSubmit);
});