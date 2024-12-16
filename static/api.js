// API related functions
const API = {
    async getSimilarNames(name) {
        try {
            const response = await fetch('http://localhost:5000/similar_names', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch similar names');
            }

            const data = await response.json();
            return data.similar_names || [];
        } catch (error) {
            throw new Error('Failed to fetch similar names. Please try again.');
        }
    }
};