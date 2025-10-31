document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        searchInput: document.getElementById('searchInput'),
        searchBtn: document.getElementById('searchBtn'),
        exportBtn: document.getElementById('exportBtn'),
        tableBody: document.getElementById('attendanceTableBody'),
        statusText: document.getElementById('statusText'),
        toastContainer: document.getElementById('notificationToast'),
        toastBody: document.querySelector('#notificationToast .toast-body'),
        startTimeForm: document.getElementById('startTimeForm'),
        classStartTime: document.getElementById('classStartTime'),
    };

    const toast = new bootstrap.Toast(elements.toastContainer);

    const fetchJSON = async (url, options = {}) => {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }
        return response.json();
    };

    const renderAttendanceRows = (records) => {
        elements.tableBody.innerHTML = '';
        const seen = new Set();
        records.forEach((record) => {
            if (seen.has(record.student_name)) {
                return;
            }
            seen.add(record.student_name);

            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${record.student_name}</td>
                <td>${record.status}</td>
                <td>${record.timestamp}</td>
                <td>${record.status === 'Late' ? '<span class="badge bg-warning">Late</span>' : ''}</td>
            `;
            elements.tableBody.appendChild(row);
        });
    };

    const loadAttendance = async (searchTerm = '') => {
        try {
            const endpoint = searchTerm ? `/api/search?term=${encodeURIComponent(searchTerm)}` : '/api/attendance';
            const data = await fetchJSON(endpoint);
            renderAttendanceRows(data);
        } catch (error) {
            console.error('Unable to load attendance data.', error);
        }
    };

    const exportAttendance = async () => {
        try {
            const response = await fetch('/api/export');
            if (!response.ok) {
                throw new Error(`Export failed with status ${response.status}`);
            }
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'attendance.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Export error:', error);
        }
    };

    const updateStartTime = async (event) => {
        event.preventDefault();
        try {
            const payload = JSON.stringify({ start_time: elements.classStartTime.value });
            const response = await fetch('/set_start_time', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
            });
            const result = await response.json();
            if (!result.success) {
                throw new Error(result.error || 'Unknown error');
            }
            alert('Class start time updated.');
        } catch (error) {
            console.error('Failed to set class start time', error);
            alert('Could not set the class start time. Please try again.');
        }
    };

    elements.searchBtn.addEventListener('click', () => {
        const term = elements.searchInput.value.trim();
        loadAttendance(term);
    });

    elements.exportBtn.addEventListener('click', (event) => {
        event.preventDefault();
        exportAttendance();
    });

    elements.startTimeForm.addEventListener('submit', updateStartTime);

    const socket = io();
    socket.on('connect', () => {
        console.log('Connected to Socket.IO server');
        elements.statusText.textContent = 'Active';
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from Socket.IO server');
        elements.statusText.textContent = 'Inactive';
    });

    socket.on('attendance_update', (data) => {
        loadAttendance(elements.searchInput.value.trim());
        elements.toastBody.textContent = `${data.student_name} marked as ${data.status} at ${data.timestamp}`;
        toast.show();
    });

    loadAttendance();
});
