; Inno Setup script: per-user installer for the PyInstaller build.
;
;   pyinstaller packaging/ChibiCompanion.spec --noconfirm
;   iscc /DAppVersion=1.0.0 packaging/installer.iss
;
; Installs into %LOCALAPPDATA%\Programs (no administrator rights needed), adds
; Start-menu and optional desktop / autostart shortcuts.  Uninstalling keeps
; the user's data (%APPDATA%\ChibiCompanion) unless they tick the box -
; memories should never be deleted by surprise.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F6B2E1A-6C1D-4C37-9C1B-8D5C2A6E0B11}
AppName=Chibi Companion
AppVersion={#AppVersion}
AppPublisher=Chibi Companion
DefaultDirName={localappdata}\Programs\ChibiCompanion
DefaultGroupName=Chibi Companion
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ChibiCompanion-Setup-{#AppVersion}
SetupIconFile=chibi.ico
UninstallDisplayIcon={app}\ChibiCompanion.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

[Languages]
Name: "ukrainian"; MessagesFile: "compiler:Languages\Ukrainian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "Запускати разом з Windows"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\ChibiCompanion\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Chibi Companion"; Filename: "{app}\ChibiCompanion.exe"
Name: "{group}\{cm:UninstallProgram,Chibi Companion}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Chibi Companion"; Filename: "{app}\ChibiCompanion.exe"; Tasks: desktopicon
Name: "{userstartup}\Chibi Companion"; Filename: "{app}\ChibiCompanion.exe"; Tasks: autostart

[Run]
Filename: "{app}\ChibiCompanion.exe"; Description: "{cm:LaunchProgram,Chibi Companion}"; Flags: nowait postinstall skipifsilent

[Code]
var
  RemoveData: Boolean;

function InitializeUninstall(): Boolean;
begin
  RemoveData := MsgBox('Видалити також усі дані Мії (пам''ять, налаштування, завантажені голосові моделі)?',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usPostUninstall) and RemoveData then
  begin
    DelTree(ExpandConstant('{userappdata}\ChibiCompanion'), True, True, True);
    DelTree(ExpandConstant('{localappdata}\ChibiCompanion'), True, True, True);
  end;
end;
