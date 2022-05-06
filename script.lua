
function main(splash, args)
    local ok, result = splash:with_timeout(function()
    --enabling the return of splash response
    splash.request_body_enabled = true
    --set your user agent
    splash:set_user_agent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36')
    splash.private_mode_enabled = false
    splash.indexeddb_enabled = true
    splash:set_viewport_full()
    splash:send_keys("<Tab>")
    splash:mouse_hover(100,100)
    splash.scroll_position = {y=200}
    --visit the given url
    local url = args.url
    local ok, reason = splash:go(url)
    if ok then
        --if no error found, wait for 1 second for the page to render
        splash:wait(1)
        --store the html content in a variable
        local content = assert(splash:html())
        --return the content
        return content
    end
end,60)

return result

end
        