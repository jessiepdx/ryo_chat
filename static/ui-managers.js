import * as DATA from "./data-managers.js"

/////////////////
// UI MANAGERS //
/////////////////

class PanelsManager {
    static instance;
    panels = {
        "left": [],
        "center": [],
        "right": []
    }
    
    panelStates = {
        "left": null,
        "center": null,
        "right": null
    }
    
    constructor() {
        if (PanelsManager.instance) {
            return PanelsManager.instance;
        }
        // Additional init of PanelsManager
        
        PanelsManager.instance = this;
    }

    addPanel(panel, position) {
        this.panels[position].unshift(panel);
    }

    showPanel(position) {
        const currentPanel = this.panelStates[position];
        const panel = this.panels[position][0];
        if (currentPanel) {
            currentPanel.hidePanel();
        }

        if (window.innerWidth < 900) {
            // hide panel on opposite side 
            if (position == "left") {
                const rightPanel = this.panels["right"][0];
                rightPanel.hidePanel();
                this.panelStates["right"] = null;
            }
            else if (position == "right") {
                const leftPanel = this.panels["left"][0];
                leftPanel.hidePanel();
                this.panelStates["left"] = null;
            }
        }

        panel.showPanel();
        this.panelStates[position] = panel;
        
        // Set UI state in local storage for persistance
        //const uiPrefs = new DATA.UIPrefs();
        //uiPrefs.updatePanelState(position, true)
    }

    hidePanel(position) {
        const currentPanel = this.panelStates[position];
        if (currentPanel) {
            currentPanel.hidePanel();
            this.panelStates[position] = null;

            // Update the UI state in local storage
            //const uiPrefs = new DATA.UIPrefs();
            //uiPrefs.updatePanelState(position, false);
        }
    }

    togglePanel(position) {
        if (position == "bottom") {
            const centerPanel = this.panels["center"][0];
            centerPanel.toggleFooter();

            return
        }
        
        const currentPanel = this.panelStates[position];
        if (currentPanel) {
            this.hidePanel(position);
        }
        else {
            this.showPanel(position);
        }
    }
}


class MenuManager {
    static instance;
    currentMenu = null;
    menus = {};
    rightPanelState = null;
    constructor() {
        if (MenuManager.instance) {
            return MenuManager.instance;
        }
        // Additional init of MenuManager
        
        MenuManager.instance = this;
    }

    // Methods
    addMenu(menuName, menuPanel) {
        this.menus[menuName] = menuPanel;
    }

    showMenu(menuName) {
        const menuPanel = this.menus[menuName];
        if (this.currentMenu) {
            // A menu is already displayed
            this.currentMenu.hidePanel();
            menuPanel.showPanel();
            this.currentMenu = menuPanel;
        }
        else {
            // Check for a right side panel and hide it
            const panelsManager = new PanelsManager();
            this.rightPanelState = panelsManager.panelStates.right
            if (this.rightPanelState) {
                this.rightPanelState.hidePanel();
            }
            // Show the menu panel
            menuPanel.showPanel();
            this.currentMenu = menuPanel;
        }
    }

    hideMenu() {
        if (this.currentMenu) {
            // A menu is already displayed
            this.currentMenu.hidePanel();
            this.currentMenu = null;

            if (this.rightPanelState) {
                this.rightPanelState.showPanel();
            }
        }
    }

    toggleMenu(menuName) {
        const menuPanel = this.menus[menuName];
        if (this.currentMenu) {
            if (this.currentMenu == menuPanel) {
                this.hideMenu();
            }
            else {
                this.showMenu(menuName);
            }
        }
        else {
            this.showMenu(menuName);
        }
    }

    buttonCallback(event) {
        let btnClicked = event.target;
        // Get the parent element until we have the parent nav
        while (btnClicked.nodeName != "NAV") {
            btnClicked = btnClicked.parentNode;
        }
        const btnName = btnClicked.id.split("-");
        const menuName = btnName[0];

        this.toggleMenu(menuName);
    }
}


class AccountManager {
    static instance;
    constructor() {
        if (AccountManager.instance) {
            return AccountManager.instance;
        }
        // Additional init of AccountManager
        
        AccountManager.instance = this;
    }

    // Methods
}


class ContentManager {
    static instance;
    constructor() {
        if (ContentManager.instance) {
            return ContentManager.instance;
        }
        // Additional init of ContentManager
        
        ContentManager.instance = this;
    }

    // Methods
}


////////////
// PANELS //
//--------//

class Panel {
    containerDOM;
    headerDOM;
    contentDOM;
    footerDOM;
    buttonDOM;
    
    constructor(containerID, buttonID = null) {
        this.containerDOM = document.getElementById(containerID);
        const header = this.containerDOM.querySelector(".panel-header");
        if (header) {
            this.headerDOM = header;
        }
        const content = this.containerDOM.querySelector(".panel-content");
        if (content) {
            this.contentDOM = content;
        }
        //console.log(content);
        const footer = this.containerDOM.querySelector(".panel-footer");
        if (footer) {
            this.footerDOM = footer;
        }
        //console.log(footer);
        if (buttonID) {
            this.buttonDOM = document.getElementById(buttonID);
        }
    }

    showFooter() {
        this.footerDOM.style.display = "flex";
    }

    hideFooter() {
        this.footerDOM.style.display = "none";
    }

    toggleFooter() {
        if (this.footerDOM.offsetParent !== null) {
            this.hideFooter();
        }
        else {
            this.showFooter();
        }
    }

    showPanel() {
        this.containerDOM.style.display = "flex";
    }

    hidePanel() {
        this.containerDOM.style.display = "none";
    }

    togglePanel() {
        if (this.containerDOM.style.display == "block" || this.containerDOM.style.display == "flex") {
            this.hidePanel();
        }
        else {
            this.showPanel();
        }
    }
}


// Menu Panels

class MenuPanel extends Panel {
    title;
    // Will update this to hold the icon svg or the DOM element
    icon; 
  
    constructor(containerID, buttonID, title, icon = true) {
      super(containerID, buttonID);
      this.title = title;
      this.icon = icon;
    }
  
    showPanel() {
      super.showPanel();
      if (this.icon) {
        this.showButton();
      }
    }
  
    hidePanel() {
      super.hidePanel();
      if (this.icon) {
        this.hideButton();
      }
    }
  
    // Adds the title alongside the existing icon in the button.
    // It assumes that the SVG with class "icon" is already present.
    showButton() {
      // Only add the title if it doesn't already exist.
      if (!this.buttonDOM.querySelector(".panel-title")) {
        const titleSpan = document.createElement("span");
        titleSpan.textContent = this.title;
        titleSpan.className = "panel-title";
        
        // Append the title after the icon. Assuming the icon is the first child.
        this.buttonDOM.appendChild(titleSpan);
      }
      // Get panel container DOM width
      const panelWidth = this.containerDOM.offsetWidth;
      // get the parent container
      const nav = this.buttonDOM.parentNode;
      nav.style.width = `calc(${panelWidth}px)`;
      this.buttonDOM.style.flexGrow = 1;
    }
  
    // Removes the title element, leaving the icon intact.
    hideButton() {
      const titleSpan = this.buttonDOM.querySelector(".panel-title");
      if (titleSpan) {
        titleSpan.remove();
      }
      const nav = this.buttonDOM.parentNode;
      nav.style.width = "auto";
      this.buttonDOM.style.flexGrow = 0;
    }
}


class AccountPanel extends MenuPanel {
    title = "Accounts";
    constructor(containerID, buttonID, title) {
        super(containerID, buttonID, title);
    }

    // Account will be unique by having some simple inputs and forms in the menu
}


class MainMenuPanel extends MenuPanel {
    title = "Main Menu";
    constructor(containerID, buttonID, title) {
        super(containerID, buttonID, title);
    }

    // Menu will be unique by having a UI Config with inputs and also sub menu
}


export { PanelsManager, MenuManager, AccountManager, ContentManager, Panel, MenuPanel, AccountPanel, MainMenuPanel }