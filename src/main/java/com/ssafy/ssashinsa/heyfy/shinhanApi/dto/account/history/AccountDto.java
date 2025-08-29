package com.ssafy.ssashinsa.heyfy.shinhanApi.dto.account.history;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccountDto {

    @JsonProperty("bankName")
    private String bankName;

    @JsonProperty("userName")
    private String userName;

    @JsonProperty("accountNo")
    private String accountNo;

    @JsonProperty("accountName")
    private String accountName;
}
