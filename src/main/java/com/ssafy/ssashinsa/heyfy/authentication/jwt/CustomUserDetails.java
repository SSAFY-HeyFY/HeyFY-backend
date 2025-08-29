package com.ssafy.ssashinsa.heyfy.authentication.jwt;

import lombok.Getter;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.User;
import java.util.Collection;

@Getter
@Setter
public class CustomUserDetails extends User {
    private String email;
    private String jti;

    public CustomUserDetails(String username, String password, String email, String jti, Collection<? extends GrantedAuthority> authorities) {
        super(username, password, authorities);
        this.email = email;
        this.jti = jti;
    }
}
