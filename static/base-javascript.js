import * as UI from "./ui-managers.js" 
import * as DATA from "./data-managers.js"

// Check the session for member data
const memberJSON = sessionStorage.getItem("member_data");
const basicMemberInfo = (memberJSON) ? JSON.parse(memberJSON) : null;

// Globals Here
const panelMan = new UI.PanelsManager();
const menuMan = new UI.MenuManager();
const member = new DATA.MemberData(basicMemberInfo);
const uiPrefs = new DATA.UIPrefs();

// Define Callback handlers
function contentLoaded() {
    // The DOM content has loaded.
    console.log("test");

    // Set up the panels
    const accountPanel = new UI.AccountPanel("account-panel", "account-btn", "Accounts");
    const centerPanel = new UI.Panel("center-panel");
    const leftPanel = new UI.Panel("left-panel");
    const mainMenuPanel = new UI.MainMenuPanel("main-menu-panel", "menu-btn", "Main Menu");
    const rightPanel = new UI.Panel("right-panel");

    menuMan.addMenu("account", accountPanel);
    menuMan.addMenu("menu", mainMenuPanel);
    panelMan.addPanel(leftPanel, "left");
    panelMan.addPanel(centerPanel, "center");
    panelMan.addPanel(rightPanel, "right");

    // Show panels based on uiPrefs
    /*
    for (const [position, state] of Object.entries(uiPrefs.panelStates)) {
        if (state) {
            panelMan.showPanel(position);
        }
    }
    */
}

// Set up Callback Listeners
document.addEventListener("DOMContentLoaded", contentLoaded);