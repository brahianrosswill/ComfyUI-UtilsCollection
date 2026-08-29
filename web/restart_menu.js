import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const restartComfyUI = async () => {
    if (!window.confirm("Restart ComfyUI?")) {
        return;
    }

    try {
        await api.fetchApi("/utils_collection/restart", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
    } catch {
        // A successful restart can close the connection before fetch resolves.
    }
};

app.registerExtension({
    name: "UtilsCollection.Restart",
    commands: [
        {
            id: "UtilsCollection.Restart",
            label: "Restart",
            menubarLabel: "Restart",
            icon: "pi pi-refresh",
            function: restartComfyUI,
        },
    ],
    menuCommands: [
        {
            path: [],
            commands: ["UtilsCollection.Restart"],
        },
    ],
});
